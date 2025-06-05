import numpy as np
import pandas as pd

import nestpy
import strax
import straxen

from ...dtypes import quanta_fields, cluster_id_fields
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

# Initialize the nestpy random generator
# The seed will be set in the compute method
nest_rng = nestpy.RandomGen.rndm()


@export
class NestYields(FuseBasePlugin):
    """Plugin that calculates the number of photons, electrons and excitons
    produced by energy deposit using nestpy."""

    __version__ = "0.2.3"

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"
    data_kind = "interactions_in_roi"

    dtype = strax.time_fields + cluster_id_fields + quanta_fields

    save_when = strax.SaveWhen.TARGET

    return_yields_only = straxen.URLConfig(
        default=False,
        type=bool,
        help="Set to True to return the yields model output directly instead of the \
        calculated actual quanta with NEST getQuanta function. Only for testing purposes.",
    )

    nest_width_parameters = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=nest_width_parameters",
        type=dict,
        help="Set to modify default NEST NRERWidthParameters to match recombination fluctuations. \
        From NEST code https://github.com/NESTCollaboration/nest/blob/v2.4.0/src/NEST.cpp \
        and NEST paper https://arxiv.org/abs/2211.10726 \
        See self.get_nest_width_parameters() for the options and default values. \
        Example use: {'fano_ER': -0.0015, 'A_ER': 0.096452}",
    )

    nest_er_yields_parameters = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=nest_er_yields_parameters",
        type=list,
        help="Set to modify default NEST ER yields parameters. Use -1 to keep default value. \
        From NEST code https://github.com/NESTCollaboration/nest/blob/v2.4.0/src/NEST.cpp \
        Used in the calcuations of BetaYieldsGR.",
    )

    fix_gamma_yield_field = straxen.URLConfig(
        default=-1.0,
        help="Field in V/cm to use for NEST gamma yield calculation. Only used if set to > 0.",
        type=float,
    )

    def setup(self):
        super().setup()

        if self.deterministic_seed or (self.user_defined_random_seed is not None):
            # Dont know but nestpy seems to have a problem with large seeds
            self.short_seed = int(repr(self.seed)[-8:])
            self.log.debug(f"Generating nest random numbers starting with seed {self.short_seed}")
        else:
            self.log.debug("Generating random numbers with seed pulled from OS")

        self.nc = nestpy.NESTcalc(nestpy.VDetector())
        self.vectorized_get_quanta = np.vectorize(self.get_quanta)
        self.updated_nest_width_parameters = self.update_nest_width_parameters()

        # Set the elements of the list so we do not run into problems with the vectorized function
        self.nest_er_yields_parameters_list = [
            float(element) for element in self.nest_er_yields_parameters
        ]

    def update_nest_width_parameters(self):

        # Get the default NEST NRERWidthsParam
        free_parameters = self.nc.default_NRERWidthsParam

        # Map the parameters names to the index in the free_parameters list
        parameters_key_map = {
            "fano_ions_NR": 0,  # Fano factor for NR Ions (default 0.4)
            "fano_excitons_NR": 1,  # Fano factor for NR Excitons (default 0.4)
            "A_NR": 2,  # A' - Amplitude for Recombinnation NR (default 0.04)
            "xi_NR": 3,  # ξ - Center for Recombination NR (default 0.50)
            "omega_NR": 4,  # ω - Width for Recombination NR (default 0.19)
            "skewness_NR": 5,  # Skewness for NR (default 2.25)
            "fano_ER": 6,  # Multiplier for Fano ER, if<0 field dep, else constant. (default 1)
            "A_ER": 7,  # A - Amplitude for Recombination ER, field dependent (default 0.096452)
            "omega_ER": 8,  # ω - Width for Recombination ER (default 0.205)
            "xi_ER": 9,  # ξ - Center for Recombination ER (default 0.45)
            "alpha_skewness_ER": 10,  # Skewness for ER (default -0.2)
        }

        if self.nest_width_parameters is not None:
            for key, value in self.nest_width_parameters.items():
                if key not in parameters_key_map:
                    raise ValueError(
                        f"Unknown NEST width parameter {key}.\
                        Available parameters: {parameters_key_map.keys()}"
                    )
                self.log.debug(f"Updating NEST width parameter {key} to {value}")
                free_parameters[parameters_key_map[key]] = value

        self.log.debug(f"Using NEST width parameters: {free_parameters}")

        return free_parameters

    def compute(self, interactions_in_roi):

        if len(interactions_in_roi) == 0:
            return np.zeros(0, dtype=self.dtype)

        # set the global nest random generator with self.short_seed
        nest_rng.set_seed(self.short_seed)
        # Now lock the seed during the computation
        nest_rng.lock_seed()
        # increment the seed. Next chunk we will use the modified seed to generate random numbers
        self.short_seed += 1

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        result["cluster_id"] = interactions_in_roi["cluster_id"]
        result["ed"] = interactions_in_roi["ed"]
        result["nestid"] = interactions_in_roi["nestid"]

        # Generate quanta:
        if len(interactions_in_roi) > 0:

            photons, electrons, excitons = self.vectorized_get_quanta(
                interactions_in_roi["ed"],
                interactions_in_roi["nestid"],
                interactions_in_roi["e_field"],
                interactions_in_roi["A"],
                interactions_in_roi["Z"],
                interactions_in_roi["create_S2"],
                interactions_in_roi["xe_density"],
            )
            result["photons"] = photons
            result["electrons"] = electrons
            result["excitons"] = excitons
        else:
            result["photons"] = np.empty(0)
            result["electrons"] = np.empty(0)
            result["excitons"] = np.empty(0)

        # Unlock the nest random generator seed again
        nest_rng.unlock_seed()

        return result

    def get_quanta(self, en, model, e_field, A, Z, create_s2, density):
        """Function to get quanta for given parameters using NEST."""

        yields_result = self.get_yields_from_NEST(en, model, e_field, A, Z, density)

        return self.process_yields(yields_result, create_s2)

    def get_yields_from_NEST(self, en, model, e_field, A, Z, density):
        """Function which uses NEST to yield photons and electrons for a given
        set of parameters."""

        # Fix for Kr83m events
        max_allowed_energy_difference = 1  # keV
        if model == 11:
            if abs(en - 32.1) < max_allowed_energy_difference:
                en = 32.1
            if abs(en - 9.4) < max_allowed_energy_difference:
                en = 9.4

        # Some additions taken from NEST code
        if model == 0 and en > 2e2:
            self.log.warning(
                f"Energy deposition of {en} keV beyond NEST validity for NR model of 200 keV"
            )
        if model == 7 and en > 3e3:
            self.log.warning(
                f"Energy deposition of {en} keV beyond NEST validity for gamma model of 3 MeV"
            )
        if model == 8 and en > 3e3:
            self.log.warning(
                f"Energy deposition of {en} keV beyond NEST validity for beta model of 3 MeV"
            )

        if model == 7 and self.fix_gamma_yield_field > 0:
            e_field = self.fix_gamma_yield_field

        if e_field < 0:
            raise ValueError(
                f"Negative electric field {e_field} V/cm not allowed. \
                (no error will be raised by NEST)."
            )

        yields_result = self.nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(model),
            energy=en,
            drift_field=e_field,
            A=A,
            Z=Z,
            density=density,
            ERYieldsParam=self.nest_er_yields_parameters_list,
        )

        return yields_result

    def process_yields(self, yields_result, create_s2):
        """Process the yields with NEST to get actual quanta."""

        # Density argument is not used in function...
        event_quanta = self.nc.GetQuanta(
            yields_result, free_parameters=self.updated_nest_width_parameters
        )

        excitons = event_quanta.excitons
        photons = event_quanta.photons
        electrons = event_quanta.electrons

        # Only for testing purposes, return the yields directly
        if self.return_yields_only:
            photons = yields_result.PhotonYield
            electrons = yields_result.ElectronYield

        # If we don't want to create S2, set electrons to 0
        if not create_s2:
            electrons = 0

        return photons, electrons, excitons


@export
class BBFYields(FuseBasePlugin):
    __version__ = "0.1.1"

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"

    dtype = quanta_fields + strax.time_fields

    def setup(self):
        super().setup()

        self.bbfyields = BBF_quanta_generator(self.rng)

    def compute(self, interactions_in_roi):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        # Generate quanta:
        if len(interactions_in_roi) > 0:
            photons, electrons, excitons = self.bbfyields.get_quanta_vectorized(
                energy=interactions_in_roi["ed"],
                interaction=interactions_in_roi["nestid"],
                field=interactions_in_roi["e_field"],
            )

            result["photons"] = photons
            result["electrons"] = electrons
            result["excitons"] = excitons
        else:
            result["photons"] = np.empty(0)
            result["electrons"] = np.empty(0)
            result["excitons"] = np.empty(0)
        return result


class BBF_quanta_generator:
    def __init__(self, rng):
        self.rng = rng
        self.er_par_dict = {
            "W": 0.013509665661431896,
            "Nex/Ni": 0.08237994367314523,
            "py0": 0.12644250072199228,
            "py1": 43.12392476032283,
            "py2": -0.30564651066249543,
            "py3": 0.937555814189728,
            "py4": 0.5864910020458629,
            "rf0": 0.029414125811261564,
            "rf1": 0.2571929264699089,
            "fano": 0.059,
        }
        self.nr_par_dict = {
            "W": 0.01374615297291325,
            "alpha": 0.9376149722771664,
            "zeta": 0.0472,
            "beta": 311.86846286764376,
            "gamma": 0.015772527423653895,
            "delta": 0.0620,
            "kappa": 0.13762801393921467,
            "eta": 6.387273512457444,
            "lambda": 1.4102590741165675,
            "fano": 0.059,
        }
        self.ERs = [7, 8, 11]
        self.NRs = [0, 1]
        self.unknown = [12]
        self.get_quanta_vectorized = np.vectorize(self.get_quanta, excluded="self")

    def update_ER_params(self, new_params):
        self.er_par_dict.update(new_params)

    def update_NR_params(self, new_params):
        self.nr_par_dict.update(new_params)

    def get_quanta(self, interaction, energy, field):
        if int(interaction) in self.ERs:
            return self.get_ER_quanta(energy, field, self.er_par_dict)
        elif int(interaction) in self.NRs:
            return self.get_NR_quanta(energy, field, self.nr_par_dict)
        elif int(interaction) in self.unknown:
            return 0, 0, 0
        else:
            raise RuntimeError(
                "Unknown nest ID: {:d}, {:s}".format(
                    int(interaction), str(nestpy.INTERACTION_TYPE(int(interaction)))
                )
            )

    def ER_recomb(self, energy, field, par_dict):
        W = par_dict["W"]
        ExIonRatio = par_dict["Nex/Ni"]

        Nq = energy / W
        Ni = Nq / (1.0 + ExIonRatio)
        Nex = Nq - Ni

        TI = par_dict["py0"] * np.exp(-energy / par_dict["py1"]) * field ** par_dict["py2"]
        Recomb = 1.0 - np.log(1.0 + TI * Ni / 4.0) / (TI * Ni / 4.0)
        FD = 1.0 / (1.0 + np.exp(-(energy - par_dict["py3"]) / par_dict["py4"]))

        return Recomb * FD

    def ER_drecomb(self, energy, par_dict):
        return par_dict["rf0"] * (1.0 - np.exp(-energy / par_dict["py1"]))

    def NR_quenching(self, energy, par_dict):
        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        gamma = par_dict["gamma"]
        delta = par_dict["delta"]
        kappa = par_dict["kappa"]
        eta = par_dict["eta"]
        lam = par_dict["lambda"]
        zeta = par_dict["zeta"]

        e = 11.5 * energy * 54.0 ** (-7.0 / 3.0)
        g = 3.0 * e**0.15 + 0.7 * e**0.6 + e

        return kappa * g / (1.0 + kappa * g)

    def NR_ExIonRatio(self, energy, field, par_dict):
        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        gamma = par_dict["gamma"]
        delta = par_dict["delta"]
        kappa = par_dict["kappa"]
        eta = par_dict["eta"]
        lam = par_dict["lambda"]
        zeta = par_dict["zeta"]

        e = 11.5 * energy * 54.0 ** (-7.0 / 3.0)

        return alpha * field ** (-zeta) * (1.0 - np.exp(-beta * e))

    def NR_Penning_quenching(self, energy, par_dict):
        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        gamma = par_dict["gamma"]
        delta = par_dict["delta"]
        kappa = par_dict["kappa"]
        eta = par_dict["eta"]
        lam = par_dict["lambda"]
        zeta = par_dict["zeta"]

        e = 11.5 * energy * 54.0 ** (-7.0 / 3.0)
        g = 3.0 * e**0.15 + 0.7 * e**0.6 + e

        return 1.0 / (1.0 + eta * e**lam)

    def NR_recomb(self, energy, field, par_dict):
        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        gamma = par_dict["gamma"]
        delta = par_dict["delta"]
        kappa = par_dict["kappa"]
        eta = par_dict["eta"]
        lam = par_dict["lambda"]
        zeta = par_dict["zeta"]

        e = 11.5 * energy * 54.0 ** (-7.0 / 3.0)
        g = 3.0 * e**0.15 + 0.7 * e**0.6 + e

        HeatQuenching = self.NR_quenching(energy, par_dict)
        PenningQuenching = self.NR_Penning_quenching(energy, par_dict)

        ExIonRatio = self.NR_ExIonRatio(energy, field, par_dict)

        xi = gamma * field ** (-delta)
        Nq = energy * HeatQuenching / par_dict["W"]
        Ni = Nq / (1.0 + ExIonRatio)

        return 1.0 - np.log(1.0 + Ni * xi) / (Ni * xi)

    def get_ER_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict["W"]
        Nq = np.clip(
            np.round(self.rng.normal(Nq_mean, np.sqrt(Nq_mean * par_dict["fano"]))), 0, np.inf
        ).astype(np.int64)

        Ni = self.rng.binomial(Nq, 1.0 / (1.0 + par_dict["Nex/Ni"]))

        recomb = self.ER_recomb(energy, field, par_dict)
        drecomb = self.ER_drecomb(energy, par_dict)
        true_recomb = np.clip(self.rng.normal(recomb, drecomb), 0.0, 1.0)

        Ne = self.rng.binomial(Ni, 1.0 - true_recomb)
        Nph = Nq - Ne
        Nex = Nq - Ni
        return Nph, Ne, Nex

    def get_NR_quanta(self, energy, field, par_dict):
        Nq_mean = energy / par_dict["W"]
        Nq = np.round(self.rng.normal(Nq_mean, np.sqrt(Nq_mean * par_dict["fano"]))).astype(
            np.int64
        )

        quenching = self.NR_quenching(energy, par_dict)
        Nq = self.rng.binomial(Nq, quenching)

        ExIonRatio = self.NR_ExIonRatio(energy, field, par_dict)
        Ni = self.rng.binomial(Nq, ExIonRatio / (1.0 + ExIonRatio))

        penning_quenching = self.NR_Penning_quenching(energy, par_dict)
        Nex = self.rng.binomial(Nq - Ni, penning_quenching)

        recomb = self.NR_recomb(energy, field, par_dict)
        if recomb < 0 or recomb > 1:
            return None, None

        Ne = self.rng.binomial(Ni, 1.0 - recomb)
        Nph = Ni + Nex - Ne
        return Nph, Ne, Nex


@export
class MigdalYields(NestYields): 
    __version__ = '0.0.1-alpha'
    child_plugin = True

    provides = ("quanta", "migdal_truth")
    
    common_dtypes = strax.time_fields + cluster_id_fields
    quanta_dtypes = common_dtypes + quanta_fields
    truth_dtype = common_dtypes + [
        (("This cluster contains a Migdal effect", "has_migdal"), np.bool_),
        (("Number of photons at interaction position caused by Migdal effect", "migdal_photons"), np.int32),
        (("Number of electrons at interaction position caused by Migdal effect", "migdal_electrons"), np.int32),
        (("Number of excitons at interaction position caused by Migdal effect", "migdal_excitons"), np.int32),
        (("Energy of Migdal electron", "migdal_energy"), np.float32),
        (("Orbital of Migdal electron", "migdal_orbital"), ">U3"),
    ] 

    dtype = {
        "quanta": quanta_dtypes,
        "migdal_truth": truth_dtype,
    }

    data_kind = {data_type: "interactions_in_roi" for data_type in provides}

    xenon_mass = straxen.URLConfig(
        default="simple_load://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=xenon_mass",
        type=(int, float),
        help="Standard atomic weight of Xenon atom in keV",
    )

    binding_energies = straxen.URLConfig(
        default="simple_load://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=xenon_binding_energies",
        type=dict,
        help="Binding energies corresponding to Xenon atomic orbitals in keV. "
        "From https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.035032",
    )

    orbitals = straxen.URLConfig(
        default="simple_load://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=considered_orbitals",
        type=list,
        help="List of orbitals to allow Migdal events from.",
    )

    distribution_manager = straxen.URLConfig(
        default = "simple_load://resource://"
        "Migdal_probability_distribution_interpolators.pkl?"
        "&fmt=pkl",
        help="Interpolators for the Migdal distribution functions. " 
        "Migdal class instance from https://github.com/petercox/Migdal/blob/v1.0.0/Migdal.py .\n",
    )

    def setup(self):
        super().setup()

        self.vectorized_get_quanta = np.vectorize(
            self.get_quanta, 
            otypes=[int, int, int, bool, int, int, int, float, str]
        )
    
    def compute(self, interactions_in_roi):

        return_empty = len(interactions_in_roi) == 0
        
        results = {}
        for data_type in self.provides:
            if return_empty:
                result = np.zeros(0, dtype=self.dtype[data_type])
            else:
                result = np.zeros(len(interactions_in_roi), dtype=self.dtype[data_type])
                result["time"] = interactions_in_roi["time"]
                result["endtime"] = interactions_in_roi["endtime"]
                result["cluster_id"] = interactions_in_roi["cluster_id"]
                result["ed"] = interactions_in_roi["ed"]
                result["nestid"] = interactions_in_roi["nestid"]
            
            results[data_type] = result

        if return_empty:
            return results
        
        # set the global nest random generator with self.short_seed
        nest_rng.set_seed(self.short_seed)
        # Now lock the seed during the computation
        nest_rng.lock_seed()
        # increment the seed. Next chunk we will use the modified seed to generate random numbers
        self.short_seed += 1

        # Generate quanta:
        if len(interactions_in_roi) > 0:

            (photons,
             electrons,
             excitons,
             has_migdal,
             migdal_photons,
             migdal_electrons,
             migdal_excitons,
             migdal_energy,
             migdal_orbital
             ) = self.vectorized_get_quanta(
                interactions_in_roi["ed"],
                interactions_in_roi["nestid"],
                interactions_in_roi["e_field"],
                interactions_in_roi["A"],
                interactions_in_roi["Z"],
                interactions_in_roi["create_S2"],
                interactions_in_roi["xe_density"],
            )
            results["quanta"]["photons"] = photons
            results["quanta"]["electrons"] = electrons
            results["quanta"]["excitons"] = excitons

            results["migdal_truth"]["has_migdal"] = has_migdal
            results["migdal_truth"]["migdal_photons"] = migdal_photons
            results["migdal_truth"]["migdal_electrons"] = migdal_electrons
            results["migdal_truth"]["migdal_excitons"] = migdal_excitons
            results["migdal_truth"]["migdal_energy"] = migdal_energy
            results["migdal_truth"]["migdal_orbital"] = migdal_orbital
        else:
            results["quanta"]["photons"] = np.empty(0)
            results["quanta"]["electrons"] = np.empty(0)
            results["quanta"]["excitons"] = np.empty(0)

            results["migdal_truth"]["has_migdal"] = np.empty(0)
            results["migdal_truth"]["migdal_photons"] = np.empty(0)
            results["migdal_truth"]["migdal_electrons"] = np.empty(0)
            results["migdal_truth"]["migdal_excitons"] = np.empty(0)
            results["migdal_truth"]["migdal_energy"] = np.empty(0)
            results["migdal_truth"]["migdal_orbital"] = np.empty(0)

        # Unlock the nest random generator seed again
        nest_rng.unlock_seed()

        return results

    def get_quanta(self, en, model, e_field, A, Z, create_s2, density):
            
        photons, electrons, excitons = super().get_quanta(en, model, e_field, A, Z, create_s2, density)
        
        # Initialise Truth variables
        has_migdal = False
        m_photons = m_electrons = m_excitons =  em_energy = 0
        orbital = None

        # If the event is a NR, add migdal
        if model == 0:
            
            erec = en 
            v = np.sqrt(2 * erec / self.xenon_mass)
    
            has_migdal, orbital = self.get_orbital(v)
            
            if has_migdal:

                binding_e = self.binding_energies[orbital]
                electron_energy = self.get_electron_energy(v, orbital)

                # TODO: Currently assuming that all of the binding energy is released as beta radiation.
                # Auger electrons and X-rays might disagree
                em_energy = electron_energy + binding_e

                m_photons, m_electrons, m_excitons = super().get_quanta(em_energy, 8, e_field, A, Z, create_s2, density)

                photons += m_photons
                excitons += m_excitons
                if create_s2:
                    electrons += m_electrons

        return photons, electrons, excitons, has_migdal, m_photons, m_electrons, m_excitons, em_energy, orbital

    def get_electron_energy(self, v, orbital):
        """
        Compute the energy of a Migdal electron based on the nucleus speed and orbital.

        This method uses the provided nucleus speed `v` and the specified orbital to 
        generate the energy of the electron by sampling from a probability distribution. 
        The probability distribution function (PDF) is computed and normalized, and 
        the inverted cumulative distribution function (CDF) is used to obtain the 
        electron energy.

        Parameters:
        - v (float): Speed of the recoiling nucleus.
        - orbital (str): Orbital shell from which the electron is ionized (e.g., '3s', '4p-').

        Returns:
        - float: Energy of the ionized electron.
        """
        es = np.logspace(-4, np.log10(20), 200)
        vs = np.repeat(v, len(es))

        # Obtain PDF
        points = self.pairwise_log_transform(es.copy(), vs.copy())
        pdf = self.distribution_manager.dpI1(points, orbital)

        # Compute CDF
        cdf = pdf.cumsum() # Not yet normalised
        cdf /= cdf[-1] # Normalised

        # Determine electron energy using inverted tranform sampling        
        random_n = self.rng.uniform()
        inverted_cdf_value = np.interp(random_n, cdf, es)

        return inverted_cdf_value

    def get_orbital(self, v):
        """
        Determine the orbital shell from which an electron is ionized, if any, based on the nucleus speed.

        This method calculates the probabilities for each orbital shell based on the nucleus 
        speed `v`. A random number is used to sample from these probabilities and determine 
        whether an electron is ionized and, if so, from which orbital shell.

        Parameters:
        - v (float): Speed of the recoiling nucleus.

        Returns:
        - Tuple[bool, str or None]: A tuple where the first element is a boolean indicating 
        whether an electron was ionized, and the second element is the orbital shell 
        ('3s', '4p', etc.) if an electron was ionized, or `None` if no electron was 
        ionized.
        """
        total_probability = 0
        probabilities = {}

        # Compute total probability of having a Migdal ionisation from considered shells
        for orbital in self.orbitals:
            _probability = self.distribution_manager.pI1(np.log(v), orbital)            
            _probability = np.where(
                np.isnan(_probability),
                0,
                _probability,
            )
            
            total_probability += _probability
            probabilities[orbital] = _probability

        random_n = self.rng.uniform()
        if random_n > total_probability:
            return False, None
        
        probabilities = pd.DataFrame(
            probabilities.values(),
            index=probabilities.keys(),
            columns=["probability"])
        probabilities['norm_probability'] = probabilities.probability / total_probability
        probabilities = probabilities.sort_values("norm_probability")

        # Determine which orbital the electron came from using inverse transform sampling
        random_n = self.rng.uniform()
        for orbital, probability in probabilities.norm_probability.cumsum().items():
            if random_n < probability:
                return True, orbital

    @staticmethod
    def pairwise_log_transform(a, b):
        """
        Applies a logarithmic transformation to two input arrays or values.

        Reshapes and concatenates the inputs into a 2D array, then computes the natural logarithm
        of each element. This ensures compatibility with functions requiring a 2D array with two columns.

        Parameters
        ----------
        a : array-like or float
            First input array or single float value.
        b : array-like or float
            Second input array or single float value.

        Returns
        -------
        numpy.ndarray
            A 2D array where each element is the natural logarithm of the input values.
        """
        a = np.atleast_1d(a).reshape(-1, 1)
        b = np.atleast_1d(b).reshape(-1, 1)
        arr = np.concatenate((a, b), axis=1)
        return np.log(arr)
import numpy as np
import nestpy
import strax
import straxen
from scipy.special import erf

from ...dtypes import quanta_fields
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

# Initialize the nestpy random generator
# The seed will be set in the compute method
nest_rng = nestpy.RandomGen.rndm()


@export
class NestYields(FuseBasePlugin):
    """Plugin that calculates the number of photons, electrons and excitons
    produced by energy deposit using nestpy."""

    __version__ = "0.3.0"

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"
    data_kind = "interactions_in_roi"

    dtype = quanta_fields + strax.time_fields

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
        type=(int, list),
        help="Set to modify default NEST ER yields parameters. Use -1 to keep default value. \
        From NEST code https://github.com/NESTCollaboration/nest/blob/v2.4.0/src/NEST.cpp \
        Used in the calcuations of BetaYieldsGR.",
    )

    nest_nr_yields_parameters = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=nest_nr_yields_parameters",
        type=(int, list),
        help="Set to modify default NEST NR yields parameters. Use -1 to keep default value.",
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

        # Change NR and ER yields to best fit:
        if hasattr(self.nest_er_yields_parameters, "__len__"):
            # Set the elements of the list
            # so we do not run into problems with the vectorized function
            self.nest_er_yields_parameters_list = [
                float(element) for element in self.nest_er_yields_parameters
            ]
        elif self.nest_er_yields_parameters == -1:
            self.nest_er_yields_parameters_list = nestpy.default_er_yields_params()
        else:
            raise ValueError("Not supported setting for nest_er_yields_parameters!")

        if hasattr(self.nest_nr_yields_parameters, "__len__"):
            self.nest_nr_yields_parameters_list = [
                float(element) for element in self.nest_nr_yields_parameters
            ]
        elif self.nest_nr_yields_parameters == -1:
            self.nest_nr_yields_parameters_list = nestpy.default_nr_yields_params()
        else:
            raise ValueError("Not supported setting for nest_nr_yields_parameters!")

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
            nuisance_parameters=self.nest_nr_yields_parameters_list,
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
class NestNRCustomERYields(NestYields):
    """Plugin that calculates the number of photons, electrons and excitons
    using NEST for NR (and any other non-ER) interactions and a custom yield
    model for ER interactions.

    The ER model is migrated from the appletree based er_nestv2_tianyu model
    (derived from NEST v2.4.0 https://github.com/NESTCollaboration/nest/releases/tag/v2.4.0
    with priors from https://arxiv.org/abs/2211.10726).
    Variables beginning with '_' are expectation values, such as `_Nph`, `_Ne`.
    """

    __version__ = "0.1.0"

    # Parameters needed by the ER yield model.
    # The field is taken per interaction.
    er_yield_model_parameter_names = (
        # Liquid xenon density [g/cm^3], used instead of the per interaction xe_density
        "liquid_xe_density",
        # Charge yield
        "m1",
        "m2",
        "m3",
        "m4",
        "m7",
        "m8",
        "m9",
        "m10",
        "w",
        # Fano factor
        "delta_f",
        # Recombination fluctuations
        "A_er",
        "B_er",
        "xi_er",
        "omega_er",
        "epsilon_er",
        "gamma_er",
        # Skewness of the electron number distribution
        "cc0",
        "cc1",
    )

    er_nestids = straxen.URLConfig(
        default=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        type=(list, tuple),
        help="NEST interaction type IDs that are handled by the custom ER yield model. \
        Default: gammaRay(7), beta(8), CH3T(9), C14(10), Kr83m(11), ppSolar(12), \
        atmNu(13), fullGamma(14), fullGamma_PE(15), fullGamma_Compton_PP(16). \
        All other IDs are passed to NEST.",
    )

    er_yield_model_parameters = straxen.URLConfig(
        # MAP values of the ER fit, SR2 values for w and liquid_xe_density
        default={
            "liquid_xe_density": 2.8787004,
            "m1": 1.1751774,
            "m2": 80.1821567,
            "m3": 0.4770064,
            "m4": 2.8309839,
            "m7": 75.0837912,
            "m8": 3.3863452,
            "m9": 0.288283,
            "m10": 0.0496054,
            "w": 0.0136321,
            "delta_f": 0.5950612,
            "A_er": 0.1082587,
            "B_er": 1.6676556,
            "xi_er": 1287.79,
            "omega_er": 3242.88,
            "epsilon_er": 0.0071204,
            "gamma_er": 1.0188102,
            "cc0": 0.2956242,
            "cc1": 3.8332914,
        },
        type=dict,
        help="Parameters of the custom ER yield model. \
        See er_yield_model_parameter_names for the required keys. \
        Example use: {'m1': 1.1751774, 'w': 0.0136321, ...}",
    )

    def setup(self):
        super().setup()
        self.er_nestids_array = np.array(self.er_nestids, dtype=int)
        self.er_par_dict = self.get_er_yield_model_parameters()

    def get_er_yield_model_parameters(self):
        """Hook to provide the ER yield model parameters as a dict.

        By default they are taken from the er_yield_model_parameters config.
        Overwrite this method to get them from somewhere else.
        """
        parameters = dict(self.er_yield_model_parameters)

        unknown = set(parameters) - set(self.er_yield_model_parameter_names)
        if unknown:
            raise ValueError(
                f"Unknown ER yield model parameters {sorted(unknown)}. \
                Available parameters: {self.er_yield_model_parameter_names}"
            )

        missing = [key for key in self.er_yield_model_parameter_names if key not in parameters]
        if missing:
            raise ValueError(f"Missing ER yield model parameters: {missing}")

        parameters = {key: float(value) for key, value in parameters.items()}
        self.log.debug(f"Using ER yield model parameters: {parameters}")

        return parameters

    def compute(self, interactions_in_roi):

        if len(interactions_in_roi) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        is_er = np.isin(interactions_in_roi["nestid"], self.er_nestids_array)

        # NR and all other interactions: NEST
        if np.any(~is_er):
            result[~is_er] = super().compute(interactions_in_roi[~is_er])

        # ER interactions: custom yield model
        if np.any(is_er):
            er_interactions = interactions_in_roi[is_er]

            if np.any(er_interactions["e_field"] < 0):
                raise ValueError("Negative electric field values are not allowed.")

            photons, electrons, excitons = self.get_er_quanta(
                er_interactions["ed"],
                er_interactions["nestid"],
                er_interactions["e_field"],
                np.full(len(er_interactions), self.er_par_dict["liquid_xe_density"]),
            )

            # If we don't want to create S2, set electrons to 0
            electrons = np.where(er_interactions["create_S2"], electrons, 0)

            result["photons"][is_er] = photons
            result["electrons"][is_er] = electrons
            result["excitons"][is_er] = excitons

        return result

    def get_er_quanta(self, energy, model, field, density):
        """Function to get quanta for ER interactions using the custom ER
        yield model.

        The model is a chain of steps, each one corresponding to a plugin of
        the original appletree model (er_nestv2_tianyu.py), executed in the
        order of their dependencies: energy -> num_photon, num_electron.

        All arguments are numpy arrays of the same length.

        Args:
            energy: deposited energy [keV]
            model: NEST interaction type ID (not used, all ER types are treated the same)
            field: electric field [V/cm]
            density: liquid xenon density [g/cm^3]

        Returns:
            num_photon, num_electron, Nex: numpy arrays
        """
        nex_ni_ratio, alf = self.exciton_ion_ratio_er(energy, density)
        charge_yield = self.qy_er(energy, nex_ni_ratio, field, density)
        light_yield = self.ly_er(charge_yield)
        _Nph, _Ne = self.mean_nph_ne(light_yield, charge_yield, energy)

        # Only for testing purposes, return the yields directly
        if self.return_yields_only:
            return _Nph, _Ne, np.zeros(len(energy))

        elecFrac, recombProb = self.mean_exciton_ion_er(nex_ni_ratio, _Nph, _Ne)
        fano_nq = self.fano_factor_er(_Nph, _Ne, field, density)
        Ni, Nex, Nq = self.true_exciton_ion_er(_Nph, _Ne, fano_nq, alf)
        Variance = self.omega_er(recombProb, Ni)
        num_photon, num_electron = self.true_photon_electron_er(
            energy, recombProb, Variance, Ni, Nq
        )

        return num_photon, num_electron, Nex

    def exciton_ion_ratio_er(self, energy, density):
        """ExcitonIonRatioER: energy -> nex_ni_ratio, alf."""
        nex_ni_ratio = (0.067366 + 0.039693 * density) * erf(energy * 0.05)
        alf = 1.0 / (1.0 + nex_ni_ratio)
        return nex_ni_ratio, alf

    def qy_er(self, energy, nex_ni_ratio, field, density):
        """QyER: energy, nex_ni_ratio -> charge_yield [electrons/keV]."""
        par = self.er_par_dict

        m1 = 30.66 + (par["m1"] - 30.66) / (1.0 + (field / 73.855) ** 2.0318) ** 0.41883
        m2 = par["m2"]
        m3 = np.log10(field) * 0.13946236 + par["m3"]
        m4 = 1.82217496 + (par["m4"] - 1.82217496) / (
            1.0 + (field / 144.65029656) ** -2.80532006
        )
        m5 = 1.0 / par["w"] / (1.0 + nex_ni_ratio) - m1
        m7 = 7.02921301 + (par["m7"] - 7.02921301) / (1.0 + (field / 256.48156448) ** 1.29119251)
        m8 = par["m8"]
        m9 = par["m9"]
        m10 = 0.0508273937 + (par["m10"] - 0.0508273937) / (
            1.0 + (field / 139.260460) ** -0.65763592
        )

        charge_yield = m1 * np.ones(len(energy))
        charge_yield += (m2 - m1) / (1 + (energy / m3) ** m4) ** m9
        charge_yield += m5
        charge_yield += -m5 / (1 + (energy / m7) ** m8) ** m10
        charge_yield = np.clip(charge_yield, 0, np.inf)

        coeff_TI = (1.0 / density) ** 0.3
        coeff_Ni = (1.0 / density) ** 1.4
        coeff_OL = (1.0 / density) ** -1.7 / np.log(1.0 + coeff_TI * coeff_Ni * density**1.7)
        charge_yield = charge_yield * (
            coeff_OL * np.log(1.0 + coeff_TI * coeff_Ni * density**1.7) * density**-1.7
        )

        return charge_yield

    def ly_er(self, charge_yield):
        """LyER: charge_yield -> light_yield [photons/keV]."""
        light_yield = 1.0 / self.er_par_dict["w"] - charge_yield
        return np.maximum(light_yield, 0.0)

    def mean_nph_ne(self, light_yield, charge_yield, energy):
        """MeanNphNe: light_yield, charge_yield, energy -> _Nph, _Ne."""
        _Nph = light_yield * energy
        _Ne = charge_yield * energy
        return _Nph, _Ne

    def mean_exciton_ion_er(self, nex_ni_ratio, _Nph, _Ne):
        """MeanExcitonIonER: nex_ni_ratio, _Nph, _Ne -> elecFrac, recombProb."""
        elecFrac = _Ne / (_Nph + _Ne)
        recombProb = 1.0 - (nex_ni_ratio + 1.0) * elecFrac
        recombProb = np.maximum(recombProb, 0.0)
        return elecFrac, recombProb

    def fano_factor_er(self, _Nph, _Ne, field, density):
        """FanoFactor: _Nph, _Ne -> fano_nq.

        Mimicking the behavior of NEST v2.4.0.
        Negative delta_f of -0.0015 restores https://arxiv.org/abs/2211.10726v3 Eq. 8
        """
        sign = np.sign(self.er_par_dict["delta_f"])
        abs_delta_f = np.abs(self.er_par_dict["delta_f"])

        # Fano factors in LXe for ER if delta_f is positive
        fano_nq = (sign + 1.0) / 2.0 * np.ones(len(_Ne)) * abs_delta_f

        # Fano factors in LXe for ER if delta_f is negative
        fano_nq_const = (
            0.12707 - 0.029623 * density - 0.0057042 * density**2 + 0.0015957 * density**3
        )
        fano_nq += (
            (1.0 - sign) / 2.0 * (fano_nq_const + abs_delta_f * np.sqrt((_Nph + _Ne) * field))
        )

        return fano_nq

    def true_exciton_ion_er(self, _Nph, _Ne, fano_nq, alf):
        """TrueExcitonIonER: _Nph, _Ne, fano_nq, alf -> Ni, Nex, Nq."""
        Nq_mean = _Nph + _Ne

        # Normal distribution truncated at 0
        Nq = np.clip(self.rng.normal(Nq_mean, np.sqrt(fano_nq * Nq_mean)), 0.0, np.inf)
        Nq = np.round(Nq).astype(np.int64)
        Ni = self.rng.binomial(Nq, alf)
        Nex = np.clip(Nq - Ni, 0, Nq)
        Ni = np.clip(Ni, 0, Nq)
        return Ni, Nex, Nq

    def omega_er(self, recombProb, Ni):
        """OmegaER: recombProb, Ni -> Variance."""
        par = self.er_par_dict

        binom = par["B_er"] * recombProb * (1 - recombProb)
        omega = par["epsilon_er"] * recombProb * (1 - recombProb)

        cntr = par["xi_er"]
        wide = par["omega_er"]
        gamm = par["gamma_er"]
        psi = par["A_er"] / (1.0 + 1.0 / (np.clip(Ni - cntr, 0, Ni) / wide) ** gamm) ** gamm

        Variance = (
            binom * Ni + (1 - recombProb) * omega * Ni**2 + (1 - recombProb) * (psi * Ni) ** 2
        )
        return Variance

    def true_photon_electron_er(self, energy, recombProb, Variance, Ni, Nq):
        """TruePhotonElectronER: energy, recombProb, Variance, Ni, Nq ->
        num_photon, num_electron."""
        par = self.er_par_dict

        E0, E1, E2, E3 = 7.7, 54, 26.7, 6.4
        skewness = 1.0 / (1.0 + np.exp((energy - E2) / E3)) * par["cc0"] * (
            1.0 - np.exp(-1.0 * energy / E0)
        ) + 1.0 / (1.0 + np.exp(-1.0 * (energy - E2) / E3)) * par["cc1"] * np.exp(
            -1.0 * energy / E1
        )
        delta = skewness / np.sqrt(1 + skewness * skewness)
        omega_corr = np.sqrt(Variance) / np.sqrt(1.0 - (2.0 / np.pi) * delta * delta)
        mu_corr = (1 - recombProb) * Ni - omega_corr * delta * np.sqrt(2.0 / np.pi)

        # Skew-normal distribution, same sampling method as appletree.randgen.skewnormal
        rvs0 = self.rng.normal(size=len(energy))
        rvs1 = self.rng.normal(size=len(energy))
        rvs = (skewness * np.abs(rvs0) + rvs1) / np.sqrt(1 + skewness**2)
        num_electron = rvs * omega_corr + mu_corr

        num_electron = np.clip(np.round(num_electron).astype(np.int64), 0, Ni)
        num_photon = np.clip(Nq - num_electron, 0, np.inf)
        return num_photon, num_electron


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

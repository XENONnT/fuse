from immutabledict import immutabledict
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, RectBivariateSpline
from typing import Tuple, Union, Dict

import nestpy
import strax
import straxen

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

    __version__ = "0.2.3"

    depends_on = ("interactions_in_roi", "electric_field_values")
    # use a tuple for type consistency with other yields plugins inheriting
    provides: Tuple[str, ...] = ("quanta",)
    data_kind: Union[str, Dict[str, str]] = "interactions_in_roi"

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
    __version__ = "0.1.0"
    child_plugin = True

    provides = ("quanta", "migdal_truth")

    quanta_dtypes = strax.time_fields + quanta_fields
    truth_dtype = strax.time_fields + [
        (("This cluster contains a Migdal effect", "has_migdal"), np.bool_),
        (
            ("Number of photons at interaction position caused by Migdal effect", "migdal_photons"),
            np.int32,
        ),
        (
            (
                "Number of electrons at interaction position caused by Migdal effect",
                "migdal_electrons",
            ),
            np.int32,
        ),
        (
            (
                "Number of excitons at interaction position caused by Migdal effect",
                "migdal_excitons",
            ),
            np.int32,
        ),
        (("Energy of Migdal electron (keV)", "migdal_electron_energy"), np.float32),
        (("Binding Energy of orbital of origin (keV)", "migdal_binding_energy"), np.float32),
        (("Total deposited ER energy (keV)", "migdal_deposited_energy"), np.float32),
        (
            (
                "Orbital of Migdal electron (first digit, n; second digit, l; sign, s=±1/2)",
                "migdal_orbital",
            ),
            np.int16,
        ),
    ]

    dtype = {
        "quanta": quanta_dtypes,
        "migdal_truth": truth_dtype,
    }

    save_when = immutabledict(
        quanta=strax.SaveWhen.ALWAYS,
        migdal_truth=strax.SaveWhen.ALWAYS,
    )

    data_kind = {data_type: "interactions_in_roi" for data_type in provides}

    xenon_mass = straxen.URLConfig(
        default=122298655.19,
        type=(int, float),
        help="Standard atomic weight of Xenon atom in keV",
    )

    xenon_binding_energies = straxen.URLConfig(
        default={
            "1s": 34.75594,
            "2s": 5.509354,
            "2p-": 5.161449,
            "2p": 4.835587,
            "3s": 1.170374,
            "3p-": 1.02478,
            "3p": 0.9612494,
            "3d-": 0.7081319,
            "3d": 0.6948998,
            "4s": 0.2293898,
            "4p-": 0.1755814,
            "4p": 0.1628001,
            "4d-": 0.07377911,
            "4d": 0.07166829,
            "5s": 0.02748725,
            "5p-": 0.01340357,
            "5p": 0.0119677,
        },
        type=dict,
        help="Binding energies corresponding to Xenon atomic orbitals in keV. "
        "From https://journals.aps.org/prd/abstract/10.1103/PhysRevD.107.035032",
    )

    xenon_orbital_encodings = straxen.URLConfig(
        default={
            "1s": 10,
            "2s": 20,
            "2p-": -21,
            "2p": 21,
            "3s": 30,
            "3p-": -31,
            "3p": 31,
            "3d-": -32,
            "3d": 32,
            "4s": 40,
            "4p-": -41,
            "4p": 41,
            "4d-": -42,
            "4d": 42,
            "5s": 50,
            "5p-": -51,
            "5p": 51,
        },
        type=dict,
        help="Orbital of Migdal electron (first digit, n; second digit, l; sign, s=±1/2)",
    )

    considered_orbitals = straxen.URLConfig(
        default=[
            "1s",
            "2s",
            "2p-",
            "2p",
            "3s",
            "3p-",
            "3p",
            "3d-",
            "3d",
            "4s",
            "4p-",
            "4p",
            "4d-",
            "4d",
            # "5s",
            # "5p-",
            # "5p",
        ],
        type=list,
        help="List of orbitals to allow Migdal events from.",
    )

    migdal_probability_tables = straxen.URLConfig(
        default="simple_load://resource://" "migdal_porbability_tables.json?" "&fmt=json",
        help="Single-ionisation dipole and exclusive probability tables"
        " from https://github.com/petercox/Migdal/blob/v1.0.0/Migdal.py .\n",
    )

    dipole = straxen.URLConfig(
        default=False,
        type=bool,
        help="Use dipole approximation probability tables instead of exclusive probability ones.",
    )

    log_scale_probability_tables = straxen.URLConfig(default=True, type=bool, help="")

    force_migdal = straxen.URLConfig(
        default=False,
        type=bool,
        help="Cause every nuclear recoil to be accompained by a Migdal ionisation.",
    )

    def setup(self):
        super().setup()

        self.distribution_manager = self.Migdal(
            self.migdal_probability_tables,
            parentClass=self,
            dipole=self.dipole,
            log_scale=self.log_scale_probability_tables,
        )

        self.vectorized_get_quanta = np.vectorize(
            self.get_quanta, otypes=[int, int, int, bool, int, int, int, float, float, float, str]
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

            (
                photons,
                electrons,
                excitons,
                has_migdal,
                migdal_photons,
                migdal_electrons,
                migdal_excitons,
                migdal_electron_energy,
                migdal_binding_energy,
                migdal_deposited_energy,
                migdal_orbital,
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
            results["migdal_truth"]["migdal_electron_energy"] = migdal_electron_energy
            results["migdal_truth"]["migdal_binding_energy"] = migdal_binding_energy
            results["migdal_truth"]["migdal_deposited_energy"] = migdal_deposited_energy
            results["migdal_truth"]["migdal_orbital"] = migdal_orbital
        else:
            results["quanta"]["photons"] = np.empty(0)
            results["quanta"]["electrons"] = np.empty(0)
            results["quanta"]["excitons"] = np.empty(0)

            results["migdal_truth"]["has_migdal"] = np.empty(0)
            results["migdal_truth"]["migdal_photons"] = np.empty(0)
            results["migdal_truth"]["migdal_electrons"] = np.empty(0)
            results["migdal_truth"]["migdal_excitons"] = np.empty(0)
            results["migdal_truth"]["migdal_electron_energy"] = np.empty(0)
            results["migdal_truth"]["migdal_binding_energy"] = np.empty(0)
            results["migdal_truth"]["migdal_deposited_energy"] = np.empty(0)
            results["migdal_truth"]["migdal_orbital"] = np.empty(0)

        # Unlock the nest random generator seed again
        nest_rng.unlock_seed()

        return results

    def get_quanta(self, en, model, e_field, A, Z, create_s2, density):

        photons, electrons, excitons = super().get_quanta(
            en, model, e_field, A, Z, create_s2, density
        )

        # Initialise Truth variables
        has_migdal = False
        m_photons = m_electrons = m_excitons = 0
        electron_energy = binding_e = em_energy = 0
        orbital = None
        orbital_encoding = 0

        # If the event is a NR, add migdal
        if model == 0:

            erec = en
            v = np.sqrt(2 * erec / self.xenon_mass)

            has_migdal, orbital = self.get_orbital(v)

            if has_migdal:

                binding_e = self.xenon_binding_energies[orbital]
                electron_energy = self.get_electron_energy(v, orbital)
                orbital_encoding = self.xenon_orbital_encodings[orbital]

                # TODO: Currently assuming that all of the binding energy
                # is released as beta radiation.
                # Auger electrons and X-rays might disagree
                em_energy = electron_energy + binding_e

                m_photons, m_electrons, m_excitons = super().get_quanta(
                    em_energy, 8, e_field, A, Z, create_s2, density
                )

                photons += m_photons
                excitons += m_excitons
                if create_s2:
                    electrons += m_electrons

        return (
            photons,
            electrons,
            excitons,
            has_migdal,
            m_photons,
            m_electrons,
            m_excitons,
            electron_energy,
            binding_e,
            em_energy,
            orbital_encoding,
        )

    def get_electron_energy(self, v, orbital):
        """Compute the energy of a Migdal electron based on the nucleus speed
        and orbital.

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
        cdf = pdf.cumsum()  # Not yet normalised
        cdf /= cdf[-1]  # Normalised

        # Determine electron energy using inverted tranform sampling
        random_n = self.rng.uniform()
        inverted_cdf_value = np.interp(random_n, cdf, es)

        return inverted_cdf_value

    def get_orbital(self, v):
        """Determine the orbital shell from which an electron is ionized, if
        any, based on the nucleus speed.

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
        for orbital in self.considered_orbitals:
            _probability = self.distribution_manager.pI1(v, orbital)
            _probability = np.where(
                np.isnan(_probability),
                0,
                _probability,
            )

            total_probability += _probability
            probabilities[orbital] = _probability

        # If force_migdal==True and if total_probability>0, force Migdal event and skip step
        if not (total_probability > 0 and self.force_migdal):
            random_n = self.rng.uniform()
            if random_n > total_probability:
                return False, None

        # Compute discrete CDFs for the different orbitals
        probabilities = pd.DataFrame(
            probabilities.values(), index=probabilities.keys(), columns=["probability"]
        )
        probabilities["norm_probability"] = probabilities.probability / total_probability
        probabilities = probabilities.sort_values("norm_probability")

        # Determine which orbital the electron came from using inverse transform sampling
        random_n = self.rng.uniform()
        for orbital, probability in probabilities.norm_probability.cumsum().items():
            if random_n < probability:
                return True, orbital

    @staticmethod
    def pairwise_log_transform(a, b):
        """Applies a logarithmic transformation to two input arrays or values.

        Reshapes and concatenates the inputs into a 2D array, then computes the natural logarithm
        of each element.
        This ensures compatibility with functions requiring a 2D array with two columns.

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

    class Migdal:
        """Class to store/access differential Migdal probabilities."""

        ##################################################
        # Nested interpolator classes
        ##################################################

        class DipoleInterpolator:
            """Callable class for dipole differential probability
            interpolation."""

            __slots__ = ("spline", "ln_v0")

            def __init__(self, ln_energies, dPdE, ln_v0):
                self.spline = CubicSpline(ln_energies, dPdE, extrapolate=False)
                self.ln_v0 = ln_v0

            def __call__(self, lnx):
                return 2 * lnx[:, 1] - 2 * self.ln_v0 + self.spline(lnx[:, 0])

        class DipoleIntegratedInterpolator:
            """Callable class for dipole integrated probability
            interpolation."""

            __slots__ = ("integral", "ln_v0")

            def __init__(self, ln_energies, energies, dPdE, ln_emin, ln_emax, ln_v0):
                integrand = energies * dPdE
                spline = CubicSpline(ln_energies, integrand)
                self.integral = np.log(spline.integrate(ln_emin, ln_emax))
                self.ln_v0 = ln_v0

            def __call__(self, lnv):
                return 2 * lnv - 2 * self.ln_v0 + self.integral

        ##################################################

        def __init__(self, migdal_probability_tables, parentClass, dipole=False, log_scale=False):
            """Initialize Migdal probability calculator.

            Parameters
            ----------
            migdal_probability_tables : dict
                Probability tables interpolate over
            parent_class : MigdalYields
                Initialised parent instance of MigdalYields
            dipole : bool
                Use dipole approximation
            log_scale : bool
                Use logarithmic scaling for exclusive probabilities
            """
            self.dipole = dipole
            self.log_scale = log_scale

            self.ORBITALS = parentClass.xenon_binding_energies

            self.emin = 1.0e-4  # keV
            self.emax = 20.0  # keV

            # Load probability tables once
            self._probability_tables = migdal_probability_tables

            # Cache for interpolators
            self._dpI1 = None
            self._pI1 = None

            # Load probabilities
            self.load_probabilities()

        ##################################################

        def dpI1(self, points, orbital=None):
            """Returns differential probability for single ionisation without
            excitation.

            Parameters
            ----------
            points : ndarray
                2D array of shape (n, 2) with log(energy) and log(velocity)
            orbital : str
                Orbital name (e.g., "1s", "2p")

            Returns
            -------
            ndarray
                Differential probabilities
            """
            if self._dpI1 is None:
                raise RuntimeError("Probabilities not loaded. Call load_probabilities() first.")

            if orbital not in self.ORBITALS:
                raise KeyError(
                    f"'{orbital}' is not a valid orbital. "
                    "Valid orbitals: {list(self.ORBITALS.keys())}"
                )

            # Handle different interpolator types
            interpolator = self._dpI1[orbital]
            if hasattr(interpolator, "ev"):  # RectBivariateSpline
                result = interpolator.ev(points[:, 0], points[:, 1])
            else:  # Custom interpolator
                result = interpolator(points)

            return np.exp(result)

        ##################################################

        def pI1(self, velocities, orbital=None):
            """Returns integrated probability for single ionisation without
            excitation.

            Parameters
            ----------
            points : ndarray
                1D or 2D array of velocity values
            orbital : str
                Orbital name

            Returns
            -------
            ndarray
                Integrated probabilities
            """
            if self._pI1 is None:
                raise RuntimeError("Probabilities not loaded. Call load_probabilities() first.")

            if orbital not in self.ORBITALS:
                raise KeyError(f"'{orbital}' is not a valid orbital.")

            v = np.asarray(velocities)
            lnv = np.log(v)

            log_p = self._pI1[orbital](lnv)
            p = np.exp(log_p)

            # Preserve scalar input -> scalar output
            if np.isscalar(velocities):
                return p.item()

            return p

        ##################################################

        def load_probabilities(self):
            """Initialize probabilities for all orbitals."""
            self._dpI1 = {}
            self._pI1 = {}

            for orbital in self.ORBITALS.keys():
                self._dpI1[orbital] = self._create_interpolator(orbital, integrated=False)
                self._pI1[orbital] = self._create_interpolator(orbital, integrated=True)

        ##################################################

        def _create_interpolator(self, orbital, integrated=False):
            """Create interpolator for a specific orbital.

            Parameters
            ----------
            orbital : str
                Orbital name
            integrated : bool
                Whether to create integrated probability interpolator

            Returns
            -------
            callable
                Interpolator function
            """
            # Determine probability type
            prob_type = "dipole" if self.dipole else "exclusive"
            if not self.dipole and self.log_scale:
                prob_type += "_log"

            # Extract data from tables
            orbital_data = self._probability_tables[prob_type][orbital]
            ne = int(orbital_data["n_energies"])
            nv = int(orbital_data["n_velocities"])

            energies = np.unique(np.array(orbital_data["energies"]))
            velocities = np.unique(np.array(orbital_data["velocities"]))
            probabilities = np.array(orbital_data["probabilities"])

            # Precompute log values
            ln_energies = np.log(energies)
            ln_velocities = np.log(velocities)
            ln_emin = np.log(self.emin)
            ln_emax = np.log(self.emax)

            if not integrated:
                return self._create_differential_interpolator(
                    ln_energies, ln_velocities, probabilities, ne, nv
                )
            else:
                return self._create_integrated_interpolator(
                    ln_energies, ln_velocities, energies, probabilities, ne, nv, ln_emin, ln_emax
                )

        ##################################################

        def _create_differential_interpolator(
            self, ln_energies, ln_velocities, probabilities, ne, nv
        ):
            """Create differential probability interpolator."""
            log_probs = np.log(probabilities)

            if self.dipole:
                return self.DipoleInterpolator(ln_energies, log_probs, ln_velocities[0])
            else:
                log_probs_2d = log_probs.reshape((ne, nv))
                return RectBivariateSpline(ln_energies, ln_velocities, log_probs_2d, kx=3, ky=3)

        ##################################################

        def _create_integrated_interpolator(
            self, ln_energies, ln_velocities, energies, probabilities, ne, nv, ln_emin, ln_emax
        ):
            """Create integrated probability interpolator."""
            if self.dipole:
                return self.DipoleIntegratedInterpolator(
                    ln_energies, energies, probabilities, ln_emin, ln_emax, ln_velocities[0]
                )
            else:
                probabilities_2d = probabilities.reshape((ne, nv))
                integral = np.zeros(nv)

                for i in range(nv):
                    integrand = energies * probabilities_2d[:, i]
                    spline = CubicSpline(ln_energies, integrand)
                    integral[i] = np.log(spline.integrate(ln_emin, ln_emax))

                return CubicSpline(ln_velocities, integral, extrapolate=False)

        ##################################################

        def get_total_probability(self, points, orbitals=None, exclude_shells=None):
            """Calculate total probability summed over orbitals.

            Parameters
            ----------
            points : ndarray
                2D array of shape (n, 2) with log(energy) and log(velocity)
            orbitals : list of str, optional
                List of orbitals to include. If None, use all orbitals.
            exclude_shells : tuple of str, optional
                Shell prefixes to exclude (e.g., ("5",) to exclude 5s, 5p)

            Returns
            -------
            ndarray
                Total probabilities
            """
            if orbitals is None:
                orbitals = self.ORBITALS.keys()

            if exclude_shells is not None:
                orbitals = [orb for orb in orbitals if not orb.startswith(exclude_shells)]

            total = np.zeros(len(points))
            for orbital in orbitals:
                total += self.dpI1(points, orbital)

            return total

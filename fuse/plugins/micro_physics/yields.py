import numpy as np
import nestpy
import strax
import straxen
import pickle

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

    __version__ = "0.2.2"

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"
    data_kind = "interactions_in_roi"

    dtype = quanta_fields + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    def setup(self):
        super().setup()

        if self.deterministic_seed or (self.user_defined_random_seed is not None):
            # Dont know but nestpy seems to have a problem with large seeds
            self.short_seed = int(repr(self.seed)[-8:])
            self.log.debug(f"Generating nest random numbers starting with seed {self.short_seed}")
        else:
            self.log.debug("Generating random numbers with seed pulled from OS")

        self.quanta_from_NEST = np.vectorize(self._quanta_from_NEST)

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

            photons, electrons, excitons = self.get_quanta(interactions_in_roi)
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

    def get_quanta(self, interactions_in_roi):

        photons, electrons, excitons = self.quanta_from_NEST(
            interactions_in_roi["ed"],
            interactions_in_roi["nestid"],
            interactions_in_roi["e_field"],
            interactions_in_roi["A"],
            interactions_in_roi["Z"],
            interactions_in_roi["create_S2"],
            log=self.log,
            density=interactions_in_roi["xe_density"],
        )

        return photons, electrons, excitons

    @staticmethod
    def _quanta_from_NEST(en, model, e_field, A, Z, create_s2, log, **kwargs):
        """Function which uses NEST to yield photons and electrons for a given
        set of parameters.

        Note:
            In case the energy deposit is outside of the range of NEST a -1
            is returned.
        Args:
            en (numpy.array): Energy deposit of the interaction [keV]
            model (numpy.array): Nest Id for qunata generation (integers)
            e_field (numpy.array): Field value in the interaction site [V/cm]
            A (numpy.array): Atomic mass number
            Z (numpy.array): Atomic number
            create_s2 (bool): Specifies if S2 can be produced by interaction,
                in this case electrons are generated.
            kwargs: Additional keyword arguments which can be taken by
                GetYields e.g. density.
        Returns:
            photons (numpy.array): Number of generated photons
            electrons (numpy.array): Number of generated electrons
            excitons (numpy.array): Number of generated excitons
        """
        nc = nestpy.NESTcalc(nestpy.VDetector())

        # Fix for Kr83m events.
        # Energies have to be very close to 32.1 keV or 9.4 keV
        # See: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L567
        # and: https://github.com/NESTCollaboration/nest/blob/master/src/NEST.cpp#L585
        max_allowed_energy_difference = 1  # keV
        if model == 11:
            if abs(en - 32.1) < max_allowed_energy_difference:
                en = 32.1
            if abs(en - 9.4) < max_allowed_energy_difference:
                en = 9.4

        # Some addition taken from
        # https://github.com/NESTCollaboration/nestpy/blob/e82c71f864d7362fee87989ed642cd875845ae3e/src/nestpy/helpers.py#L94-L100
        if model == 0 and en > 2e2:
            log.warning(
                f"Energy deposition of {en} keV beyond NEST validity " "for NR model of 200 keV"
            )

        if model == 7 and en > 3e3:
            log.warning(
                f"Energy deposition of {en} keV beyond NEST validity " "for gamma model of 3 MeV"
            )

        if model == 8 and en > 3e3:
            log.warning(
                f"Energy deposition of {en} keV beyond NEST validity " "for beta model of 3 MeV"
            )

        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(model),
            energy=en,
            drift_field=e_field,
            A=A,
            Z=Z,
            **kwargs,
        )

        event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...

        photons = event_quanta.photons
        excitons = event_quanta.excitons
        electrons = 0
        if create_s2:
            electrons = event_quanta.electrons

        return photons, electrons, excitons


@export
class BetaYields(NestYields):
    """Plugin that calculates the number of photons, electrons and excitons
    produced by energy deposit using nestpy."""

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"
    data_kind = "interactions_in_roi"

    # Config options

    use_recombination_fluctuation = straxen.URLConfig(
        default=True,
        type=bool,
        help="Turn on or off the recombination fluctuation for beta interactions.",
    )

    recombination_fluctuation_std_factor = straxen.URLConfig(
        default=3,
        type=(int, float),
        help="A factor that is defined to guess the recombination fluctuation",
    )

    beta_quanta_spline = straxen.URLConfig(
        default=None,
        help="Path to function that gives n_ph and n_e for a given energy, \
        calculated from beta spectrum. The function should be a pickle file.",
    )

    def get_quanta(self, interactions_in_roi):

        # for the non beta interactions we use the nestpy yields
        photons, electrons, excitons = self.quanta_from_NEST(
            interactions_in_roi["ed"],
            interactions_in_roi["nestid"],
            interactions_in_roi["e_field"],
            interactions_in_roi["A"],
            interactions_in_roi["Z"],
            interactions_in_roi["create_S2"],
            log=self.log,
            density=interactions_in_roi["xe_density"],
        )

        mask_beta = interactions_in_roi["nestid"] == 8

        # now for the beta interactions we use the beta yields
        photons_beta, electrons_beta = self.quanta_from_spline(
            interactions_in_roi["ed"][mask_beta],
        )

        photons[mask_beta] = photons_beta
        electrons[mask_beta] = electrons_beta

        return photons, electrons, excitons

    def quanta_from_spline(self, energy):

        with open(self.beta_quanta_spline, "rb") as f:
            cs1_poly, cs2_poly = pickle.load(f)

        beta_photons = cs1_poly(energy)
        beta_electrons = cs2_poly(energy)

        if self.use_recombination_fluctuation:

            factor = self.recombination_fluctuation_std_factor
            rf = self.rng.normal(0, energy * factor, len(energy))
            beta_photons += rf
            beta_electrons -= rf

            # make integer quanta
            beta_photons = np.round(beta_photons).astype(int)
            beta_electrons = np.round(beta_electrons).astype(int)

            # make sure we don't have negative quanta, so clip at 0
            beta_photons = np.clip(beta_photons, 0, np.inf)
            beta_electrons = np.clip(beta_electrons, 0, np.inf)

        return beta_photons, beta_electrons


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

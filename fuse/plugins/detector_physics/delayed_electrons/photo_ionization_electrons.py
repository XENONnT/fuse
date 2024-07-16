import strax
import straxen
import numpy as np
from scipy.stats import truncexpon

from ....plugin import FuseBasePlugin

export, __all__ = strax.exporter()


@export
class PhotoIonizationElectrons(FuseBasePlugin):
    """Plugin to simulate the emission of delayed electrons from
    photoionization in the liquid xenon using a phenomenological model.

    The plugin uses the number of S2 photons per energy deposit as input
    and creates delayed_interactions_in_roi. The simulation of delayed
    electrons can be enabled or disabled using the config option
    enable_delayed_electrons. The amount of delayed electrons can be
    scaled using the config option photoionization_modifier.
    """

    __version__ = "0.0.2"

    depends_on = (
        "s2_photons_sum",
        "extracted_electrons",
        "s2_photons",
        "electron_time",
        "microphysics_summary",
    )
    provides = "photo_ionization_electrons"
    data_kind = "delayed_interactions_in_roi"

    save_when = strax.SaveWhen.ALWAYS

    # Not start at 0. 0 are set per default for contributing clusters so we want to avoid that
    delayed_cluster_index = -1

    # Config options

    enable_delayed_electrons = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help="Decide if you want to to enable delayed electrons from photoionization",
    )

    # Calculate this from TPC dimenstions and drift velocity
    photoionization_time_cutoff = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=photoionization_time_cutoff",
        type=(int, float),
        cache=True,
        help="Time window for photoionization after a S2 in [ns]",
    )

    # Add this to our simulation config
    photoionization_time_constant = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=photoionization_time_constant",
        type=(int, float),
        cache=True,
        help="Timeconstant for photoionization in [ns]",
    )

    photoionization_modifier = straxen.URLConfig(
        default="xedocs://photoionization_strengths?version=v2&run_id=plugin.run_id&attr=value",
        type=(int, float),
        cache=True,
        help="Photoionization modifier",
    )

    diffusion_constant_longitudinal = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=diffusion_constant_longitudinal",
        type=(int, float),
        cache=True,
        help="Longitudinal electron drift diffusion constant",
    )

    drift_velocity_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_velocity_liquid",
        type=(int, float),
        cache=True,
        help="Drift velocity of electrons in the liquid xenon",
    )

    drift_time_gate = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=drift_time_gate",
        type=(int, float),
        cache=True,
        help="Electron drift time from the gate in ns",
    )

    tpc_radius = straxen.URLConfig(
        default="take://resource://" "SIMULATION_CONFIG_FILE.json?&fmt=json" "&take=tpc_radius",
        type=(int, float),
        cache=True,
        help="Radius of the XENONnT TPC [cm]",
    )

    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain",
    )

    # Is this the correct value?
    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    electron_extraction_yield = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_extraction_yield",
        type=(int, float),
        cache=True,
        help="Electron extraction yield",
    )

    def infer_dtype(self):
        # Thake the same dtype as microphysics_summary
        dtype = self.deps["s2_photons"].deps["microphysics_summary"].dtype

        return dtype

    def setup(self):
        super().setup()

        self.photoionization_scaling = (
            self.s2_secondary_sc_gain_mc * self.electron_extraction_yield
        ) / (1 + self.p_double_pe_emision)

        self.photoionization_cutoff = (
            self.photoionization_time_cutoff / self.photoionization_time_constant
        )

    def compute(self, interactions_in_roi, individual_electrons):

        # Just apply this to clusters with S2 photons
        mask = interactions_in_roi["sum_s2_photons"] > 0

        if not self.enable_delayed_electrons or (len(interactions_in_roi[mask]) == 0):
            self.log.debug(
                "No interactions with S2 photons found or delayed electrons are disabled"
            )
            return np.zeros(0, self.dtype)

        electrons_per_interaction, unique_cluster_id = group_electrons_by_cluster_id(
            individual_electrons
        )
        matching_index = np.searchsorted(unique_cluster_id, interactions_in_roi[mask]["cluster_id"])

        # In WFSim the part is calculated separatley for each interaction
        # We can do it vectorized!
        n_delayed_electrons = self.rng.poisson(
            interactions_in_roi[mask]["sum_s2_photons"]
            * self.photoionization_modifier
            / self.photoionization_scaling
        )

        electron_delay = truncexpon.rvs(
            self.photoionization_cutoff,
            size=np.sum(n_delayed_electrons),
            scale=self.photoionization_time_constant,
            random_state=self.rng,
        )

        electron_delay = np.split(electron_delay, np.cumsum(n_delayed_electrons)[:-1])

        # Randomly select the time of the extracted electrons as time zeros
        # This differs to the WFSim implementation but neglecting the photon
        # propagation time should not do much i guess
        time_zero = []
        delayed_electrons_per_interaction = []

        for i in range(len(interactions_in_roi[mask])):
            electrons_of_interaction = electrons_per_interaction[matching_index[i]]
            number_of_delayed_electrons = len(electron_delay[i])

            time_zero.append(
                self.rng.choice(electrons_of_interaction["time"], size=number_of_delayed_electrons)
            )
            delayed_electrons_per_interaction.append(number_of_delayed_electrons)

        delayed_electrons_per_interaction = np.array(delayed_electrons_per_interaction)
        time_zero = np.concatenate(time_zero)
        electron_delay = np.concatenate(electron_delay)
        n_instruction = len(electron_delay)

        result = np.zeros(n_instruction, dtype=self.dtype)
        result["time"] = time_zero  # WFsim subtracts the drift time here, i guess we dont need it??
        result["endtime"] = time_zero
        result["x"], result["y"] = ramdom_xy_position(n_instruction, self.tpc_radius, self.rng)
        result["z"] = -electron_delay * self.drift_velocity_liquid
        result["electrons"] = [1] * n_instruction

        result["cluster_id"] = self.delayed_cluster_index - np.arange(n_instruction) - 1
        self.delayed_cluster_index = np.min(result["cluster_id"])

        return strax.sort_by_time(result)


def ramdom_xy_position(n, radius, rng):
    """Generate random x and y positions for n electrons."""
    r = np.sqrt(rng.uniform(0, radius * radius, n))
    angle = rng.uniform(-np.pi, np.pi, n)

    return r * np.cos(angle), r * np.sin(angle)


# This sort of groupby function is used in several places in fuse
# We should try to make it a general function in the future.
def group_electrons_by_cluster_id(electrons):
    """Function to group electrons by cluster_id."""

    sort_index = np.argsort(electrons["cluster_id"])

    electrons_sorted = electrons[sort_index]

    unique_cluster_id, split_position = np.unique(electrons_sorted["cluster_id"], return_index=True)
    return np.split(electrons_sorted, split_position[1:]), unique_cluster_id

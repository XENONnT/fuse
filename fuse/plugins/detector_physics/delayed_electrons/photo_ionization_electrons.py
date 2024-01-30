# Add instruction generation for photo ionization electrons here

import strax
import straxen
import numpy as np
import logging

from ....common import FUSE_PLUGIN_TIMEOUT

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.delayed_electrons.photo_ionization_electrons')

@export
class PhotoIonizationElectrons(strax.Plugin):

    __version__ = "0.0.1"

    #Try to build these ones from the SecondaryScintillation output first
    # We are now having the number of photons of an interaction as input
    # In WFSim the number of photons in a potentially merged S2 is used.
    depends_on = ("s2_photons_sum", "extracted_electrons", "s2_photons", "electron_time", "microphysics_summary")
    provides = "photo_ionization_electrons"
    data_kind = "delayed_interactions_in_roi"

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.ALWAYS

    input_timeout = FUSE_PLUGIN_TIMEOUT

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    enable_delayed_electrons = straxen.URLConfig(
        default=False, type=bool, track=True,
        help='Decide if you want to to enable delayed electrons from photoionization',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    #Move the filename to the config file
    delaytime_pmf_hist = straxen.URLConfig(
        help='delaytime_pmf_hist',
        default = 'simple_load://resource://format://XENONnT_SR1_photoionization_model.dill?&fmt=dill',
    )

    photoionization_modifier = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=photoionization_modifier",
        type=(int, float),
        cache=True,
        help='Photoionization modifier',
    )

    diffusion_constant_longitudinal = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=diffusion_constant_longitudinal",
        type=(int, float),
        cache=True,
        help='Longitudinal electron drift diffusion constant',
    )

    drift_velocity_liquid = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=drift_velocity_liquid",
        type=(int, float),
        cache=True,
        help='Drift velocity of electrons in the liquid xenon',
    )

    tpc_radius = straxen.URLConfig(
        default = "take://resource://"
                  "SIMULATION_CONFIG_FILE.json?&fmt=json"
                  "&take=tpc_radius",
        type=(int, float),
        cache=True,
        help='Radius of the XENONnT TPC ',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running PhotoIonizationElectrons in debug mode")
        else: 
            log.setLevel('WARNING')

        if self.deterministic_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed = seed)
            log.debug(f"Generating random numbers from seed {seed}")
        else: 
            self.rng = np.random.default_rng()
            log.debug(f"Generating random numbers with seed pulled from OS")

    def infer_dtype(self):
        #Thake the same dtype as microphysics_summary
        dtype = self.deps["s2_photons"].deps["microphysics_summary"].dtype

        return dtype

    def compute(self, interactions_in_roi, individual_electrons):

        #Just apply this to clusters with S2 photons
        mask = interactions_in_roi["sum_s2_photons"] > 0

        if (len(interactions_in_roi[mask]) == 0) or (self.enable_delayed_electrons == False):
            log.debug("No interactions with S2 photons found or delayed electrons are disabled")
            return np.zeros(0, self.dtype)

        electrons_per_interaction, unique_cluster_id = group_electrons_by_cluster_id(individual_electrons)

        #In WFSim the part is calculated separatley for each interaction
        #We can do it vectorized!
        n_delayed_electrons = self.rng.poisson(
            interactions_in_roi[mask]["sum_s2_photons"]
            * self.photoionization_modifier
            )

        electron_delay = get_random(self.delaytime_pmf_hist, self.rng, np.sum(n_delayed_electrons))
        electron_delay = np.split(electron_delay, np.cumsum(n_delayed_electrons)[:-1])

        #Randomly select the time of the extracted electrons as time zeros
        #This differs to the WFSim implementation but neglecting the photon 
        #propagation time should not do much i guess
        time_zero = []
        delayed_electrons_per_interaction = []

        for i in range(len(interactions_in_roi[mask])):
            electrons_of_interaction = electrons_per_interaction[np.argwhere(interactions_in_roi[mask]["cluster_id"][i] == unique_cluster_id)[0][0]]
            number_of_delayed_electrons = len(electron_delay[i])
            
            time_zero.append(self.rng.choice(electrons_of_interaction["time"], size = number_of_delayed_electrons))
            delayed_electrons_per_interaction.append(number_of_delayed_electrons)

        delayed_electrons_per_interaction = np.array(delayed_electrons_per_interaction)
        time_zero = np.concatenate(time_zero)
        electron_delay = np.concatenate(electron_delay)
        n_instruction = len(electron_delay)

        result = np.zeros(n_instruction, dtype = self.dtype)
        result["time"] = time_zero #WFsim subtracts the drift time here, i guess we dont need it??
        result["endtime"] = time_zero
        result["x"], result["y"] = ramdom_xy_position(
            n_instruction,
            self.tpc_radius,
            self.rng
            )
        result['z'] = - electron_delay * self.drift_velocity_liquid
        result['electrons'] = [1]*n_instruction
        #result['cluster_id'] = np.repeat(interactions_in_roi[mask]["cluster_id"], delayed_electrons_per_interaction)
        result['cluster_id'] = np.arange(len(result)) * -1 - 1 #Lets try to use negative cluster ids for delayed electrons...
        
        return strax.sort_by_time(result)


def ramdom_xy_position(n, radius, rng):
    """Generate random x and y positions for n electrons"""
    r = np.sqrt(rng.uniform(0, radius*radius, n))
    angle = rng.uniform(-np.pi, np.pi, n)

    return r * np.cos(angle), r * np.sin(angle)

def get_random(hist_object,rng ,size=10):
    """
    get_random function from multihist but with fixed seed for reproducibility
    https://github.com/JelleAalbers/multihist/blob/6c4f786bc95f0e73ffec228e17c957744e0d9594/multihist.py#L186
    """
    bin_i = rng.choice(np.arange(len(hist_object.bin_centers)), size=size, p=hist_object.normalized_histogram)
    return hist_object.bin_centers[bin_i] + rng.uniform(-0.5, 0.5, size=size) * hist_object.bin_volumes()[bin_i]

#This sort of groupby function is used in several places in fuse
#We should try to make it a general function in the future. 
def group_electrons_by_cluster_id(electrons):
    """Function to group electrons by cluster_id"""
    
    sort_index = np.argsort(electrons["cluster_id"])
    
    electrons_sorted = electrons[sort_index]

    unique_cluster_id, split_position = np.unique(electrons_sorted["cluster_id"], return_index=True)
    return np.split(electrons_sorted, split_position[1:]), unique_cluster_id
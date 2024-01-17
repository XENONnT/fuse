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
    # In WFSim the number of photons in a potentially merged S2 is used... 
    # The wfsim approach is more difficult to include in fuse at the moment...
    # And this way we should get good results too 
    depends_on = ("s2_photons_sum", "extracted_electrons", "s2_photons")
    provides = "photo_ionization_electrons"
    data_kind = "delayed_interactions_in_roi"

    #dtype will be the same as the microphysics_summary
    #We can probably import it from there
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int64),
             ('create_S2', np.bool8),
            ]
    
    dtype += [('e_field', np.int64),
              ]
    
    dtype += [('photons', np.float64),
              ('electrons', np.float64),
              ('excitons', np.float64),
             ]
    
    dtype = dtype + strax.time_fields

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

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
        default = 'simple_load://resource://format://xnt_se_afterpulse_delaytime.pkl.gz?&fmt=pkl.gz',
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

    def compute(self, interactions_in_roi, individual_electrons):

        #Just apply this to clusters with S2 photons
        mask = interactions_in_roi["sum_s2_photons"] > 0

        if (len(interactions_in_roi[mask]) == 0) or (self.enable_delayed_electrons == False):
            log.debug("No interactions with S2 photons found or delayed electrons are disabled")
            return np.zeros(0, self.dtype)

        electrons_per_interaction = np.split(individual_electrons, np.cumsum(interactions_in_roi[mask]["n_electron_extracted"]))[:-1]

        #In WFSim the part is calculated separatley for each interaction
        #We can do it vectorized!
        n_delayed_electrons = self.rng.poisson(
            interactions_in_roi[mask]["sum_s2_photons"]
            * self.delaytime_pmf_hist.n
            * self.photoionization_modifier
            )

        electron_delay = get_random(self.delaytime_pmf_hist, self.rng, np.sum(n_delayed_electrons))
        electron_delay = np.split(electron_delay, np.cumsum(n_delayed_electrons)[:-1])

        #Randomly select the time of the extracted electrons as time zeros
        #This differs to the WFSim implementation but neglecting the photon 
        #propagation time should not do much i guess
        t_zeros = [self.rng.choice(electrons["time"], size = len(delayed_electrons)) for electrons, delayed_electrons in zip(electrons_per_interaction, electron_delay)]
        t_zeros = np.concatenate(t_zeros)
        electron_delay = np.concatenate(electron_delay)
        n_instruction = len(electron_delay)

        result = np.zeros(n_instruction, dtype = self.dtype)
        result["time"] = t_zeros #WFsim subtracts the drift time here, i guess we dont need it??
        result["endtime"] = t_zeros
        result["x"], result["y"] = ramdom_xy_position(
            n_instruction,
            self.tpc_radius,
            self.rng
            )
        result['z'] = - electron_delay * self.drift_velocity_liquid
        result['electrons'] = [1]*n_instruction
        
        return result


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
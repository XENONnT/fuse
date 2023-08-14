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

    __version__ = "0.0.0"

    #Try to build these ones from the SecondaryScintillation output first
    # We are now having the number of photons of an interaction as input
    # In WFSim the number of photons in a potentially merged S2 is used... 
    # The wfsim approach is more difficult to include in fuse at the moment...
    # And this way we should get good results too 
    depends_on = ("s2_photons_sum", "extracted_electrons", "s2_photons")
    provides = "photo_ionization_electrons"
    data_kind = "interactions_in_roi"

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

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    delaytime_pmf_hist = straxen.URLConfig(
        help='delaytime_pmf_hist',
    )

    photoionization_modifier = straxen.URLConfig(
        type=(int, float),
        help='photoionization_modifier',
    )

    diffusion_constant_longitudinal = straxen.URLConfig(
        type=(int, float),
        help='diffusion_constant_longitudinal',
    )

    drift_velocity_liquid = straxen.URLConfig(
        type=(int, float),
        help='drift_velocity_liquid',
    )

    tpc_radius = straxen.URLConfig(
        type=(int, float),
        help='tpc_radius',
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

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, self.dtype)

        electrons_per_interaction = np.split(individual_electrons, np.cumsum(interactions_in_roi[mask]["n_electron_extracted"]))[:-1]

        #In WFSim the part is calculated separatley for each interaction
        #We can do it vectorized!
        n_delayed_electrons = self.rng.poisson(
            interactions_in_roi[mask]["sum_s2_photons"]
            * self.delaytime_pmf_hist.n
            * self.photoionization_modifier
            )

        output = []
        #Maybe we can vectorize this too?
        for i in range(len(n_delayed_electrons)):

            electron_delay = get_random(self.delaytime_pmf_hist, self.rng, n_delayed_electrons[i])
        

            ### What is this function doing?? It is super slow, do i need this at all???
            # Reasonably bin delay time that would be diffuse out together
            #electron_delay_i, n_delayed_electron_i = self._reduce_instruction_timing(
            #    electron_delay,
            #    self.delaytime_pmf_hist
            #    )
            electron_delay_i = electron_delay[i]
            n_delayed_electron_i = n_delayed_electrons[i]
            n_instruction = len(electron_delay_i)

            #Randomly select the time of the extracted electrons as time zeros
            #This differs to the WFSim implementation but the effect should be small
            t_zeros = electrons_per_interaction[i]["time"][self.rng.integers(low = 0, high = len(electrons_per_interaction[i]), size = n_instruction)]

            #And build the output
            temp_output = np.zeros(n_instruction, dtype = self.dtype)
            temp_output["time"] = t_zeros #WFsim subtracts the drift time here, i guess we dont need it??
            temp_output["x"], temp_output["y"] = ramdom_xy_position(
                                                    n_instruction,
                                                    self.tpc_radius,
                                                    self.rng
                                                    )
            temp_output['z'] = - electron_delay_i * self.drift_velocity_liquid
            temp_output['electrons'] = n_delayed_electron_i
            output.append(temp_output)

        return np.concatenate(output)


    def _reduce_instruction_timing(self, electron_delay, delaytime_pmf_hist):
        # Binning the delay time, so electron timing within each
        # will be diffused to fill the whole bin
        
        delaytime_spread = np.sqrt(2 * self.diffusion_constant_longitudinal\
                                   * delaytime_pmf_hist.bin_centers)
        delaytime_spread /= self.drift_velocity_liquid

        coarse_time, coarse_time_i = [], 100 # Start at 100ns, as its smaller than single electron width
        while coarse_time_i < delaytime_pmf_hist.bin_centers[-1]:
            coarse_time.append(coarse_time_i)
            coarse_time_i += delaytime_spread[np.argmin(np.abs(coarse_time_i - delaytime_pmf_hist.bin_centers))]
        coarse_time = np.array(coarse_time)

        idx = np.digitize(electron_delay[electron_delay < coarse_time[-1]], coarse_time)
        idxs, n = np.unique(idx, return_counts=True)
        _ap_delay = coarse_time[idxs]
        return _ap_delay, n


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
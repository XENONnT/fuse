# Add instruction generation for photo electric electrons here
#Can be removed i think- we just want the photo ionization ones!

import strax
import straxen
import numpy as np
import logging

from ....common import FUSE_PLUGIN_TIMEOUT

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.delayed_electrons.photo_electric_electrons')

@export
class PhotoElectricElectrons(strax.Plugin):

    __version__ = "0.0.0"

    #Try to build these ones from the SecondaryScintillation output first
    # We are now having the number of photons of an interaction as input
    # In WFSim the number of photons in a potentially merged S2 is used... 
    # The wfsim approach is more difficult to include in fuse at the moment...
    depends_on = ("s2_photons_sum")
    provides = "photo_electric_electrons"
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

    #Can we please rename them to something more self-explanatory?
    # p for probability?
    photoelectric_p = straxen.URLConfig(
        type=(int, float),
        help='photoelectric_p',
    )

    photoelectric_modifier = straxen.URLConfig(
        type=(int, float),
        help='photoelectric_modifier',
    )

    photoelectric_t_center = straxen.URLConfig(
        type=(int, float),
        help='photoelectric_t_center',
    )

    drift_time_gate = straxen.URLConfig(
        type=(int, float),
        help='drift_time_gate',
    )

    photoelectric_t_spread = straxen.URLConfig(
        type=(int, float),
        help='photoelectric_t_spread',
    )

    drift_time_gate = straxen.URLConfig(
        type=(int, float),
        help='drift_time_gate',
    )

    tpc_radius = straxen.URLConfig(
        type=(int, float),
        help='tpc_radius',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectronDrift in debug mode")
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


    def compute(self, interactions_in_roi):

        #Just apply this to clusters with S2 photons
        mask = interactions_in_roi["sum_s2_photons"] > 0

        #In WFSim the part is calculated separatley for each interaction
        #We can do it vectorized!
        n_delayed_electrons = self.rng.poisson(
            interactions_in_roi[mask]["sum_s2_photons"]
            * self.photoelectric_p
            * self.photoelectric_modifier
            )
        
        #This will be converted to the z position of the individual electrons
        time_delayed_electrons = np.clip(
            self.rng.normal(loc=self.photoelectric_t_center+ self.drift_time_gate, 
                             scale=self.photoelectric_t_spread,
                             size=np.sum(n_delayed_electrons)),
            0, None
            )
        
        #We need to add the time when the electrons are started
        #We would usually using one of the propagated phton times
        #For now let's just use the time of the interaction
        #Fix this later when the other plugins are ready
        time_zeros = interactions_in_roi[mask]["time"] + self.drift_time_gate

        result = np.zeros(len(time_delayed_electrons), dtype=self.dtype)
        result['time'] = time_zeros
        result['x'], result['y'] = ramdom_xy_position(
            np.sum(n_delayed_electrons),
            self.tpc_radius,
            self.rng
            )
        result['z'] = - time_delayed_electrons * self.config['drift_velocity_liquid']
        result['electrons'] = 1

def ramdom_xy_position(n, radius, rng):
    """Generate random x and y positions for n electrons"""
    r = np.sqrt(rng.uniform(0, radius*radius, n))
    angle = rng.uniform(-np.pi, np.pi, n)

    return r * np.cos(angle), r * np.sin(angle)
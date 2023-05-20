import strax
import straxen
import logging

import numpy as np

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.s1_photon_hits')
log.setLevel('WARNING')

@export
class S1PhotonHits(strax.Plugin):

    __version__ = '0.0.0'

    depends_on = ("microphysics_summary")
    provides = "s1_photons"
    data_kind = "interactions_in_roi"

    #Forbid rechunking
    rechunk_on_save = False

    dtype = [('n_photon_hits', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,
        help='Show debug informations',
    )

    s1_lce_correction_map = straxen.URLConfig(
        cache=True,
        help='s1_lce_correction_map',
    )

    p_double_pe_emision = straxen.URLConfig(
        type=(int, float),
        help='p_double_pe_emision',
    )

    s1_detection_efficiency = straxen.URLConfig(
        type=(int, float),
        help='Some placeholder for s1_detection_efficiency',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running S1PhotonHits in debug mode")

    def compute(self, interactions_in_roi):

        #Just apply this to clusters with photons
        mask = interactions_in_roi["photons"] > 0

        if len(interactions_in_roi[mask]) == 0:
            return np.zeros(0, self.dtype)
        
        x = interactions_in_roi[mask]['x']
        y = interactions_in_roi[mask]['y']
        z = interactions_in_roi[mask]['z']
        n_photons = interactions_in_roi[mask]['photons'].astype(np.int64)

        positions = np.array([x, y, z]).T

        n_photon_hits = self.get_n_photons(n_photons=n_photons,
                                           positions=positions,
                                           )
        
        result = np.zeros(interactions_in_roi.shape[0], dtype = self.dtype)

        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        result["n_photon_hits"][mask] = n_photon_hits

        return result
    

    def get_n_photons(self, n_photons, positions):
    
        """Calculates number of detected photons based on number of photons in total and the positions
        :param n_photons: 1d array of ints with number of emitted S1 photons:
        :param positions: 2d array with xyz positions of interactions
        :param s1_lce_correction_map: interpolator instance of s1 light yield map
        :param config: dict wfsim config 

        return array with number photons"""
        ly = self.s1_lce_correction_map(positions)
        # depending on if you use the data driven or mc pattern map for light yield 
        #the shape of n_photon_hits will change. Mc needs a squeeze
        if len(ly.shape) != 1:
            ly = np.squeeze(ly, axis=-1)
        ly /= 1 + self.p_double_pe_emision
        ly *= self.s1_detection_efficiency

        n_photon_hits = np.random.binomial(n=n_photons, p=ly)

        return n_photon_hits
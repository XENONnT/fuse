import numpy as np
import strax
import straxen
import logging

from strax import deterministic_hash
from scipy.interpolate import interp1d

from ...common import loop_uniform_to_pe_arr

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.photon_propagation_base')
log.setLevel('WARNING')

@export
class PhotonPropagationBase(strax.Plugin):

    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

    #dtype is the same for S1 and S2
    dtype = [('channel', np.int64),
             ('dpe', np.bool_),
             ('photon_gain', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Config options shared by S1 and S2 simulation 
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    p_double_pe_emision = straxen.URLConfig(
        type=(int, float),
        help='p_double_pe_emision',
    )

    pmt_transit_time_spread = straxen.URLConfig(
        type=(int, float),
        help='pmt_transit_time_spread',
    )

    pmt_transit_time_mean = straxen.URLConfig(
        type=(int, float),
        help='pmt_transit_time_mean',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        type=(int, float),
        help='pmt_circuit_load_resistor',
    )

    digitizer_bits = straxen.URLConfig(
        type=(int, float),
        help='digitizer_bits',
    )

    digitizer_voltage_range = straxen.URLConfig(
        type=(int, float),
        help='digitizer_voltage_range',
    )

    n_top_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs on top array',
    )

    n_tpc_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs in the TPC',
    )

    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )

    photon_area_distribution = straxen.URLConfig(
        cache=True,
        help='photon_area_distribution',
    )

    def setup(self):

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]
        
        #I dont like this part -> clean up before merging the PR
        self._cached_uniform_to_pe_arr = {}
        self.__uniform_to_pe_arr = self.init_spe_scaling_factor_distributions()

    def compute(self):
        """
        Needs to be set in the child plugins!
        """
        raise NotImplementedError

    def pmt_transition_time_spread(self, _photon_timings, _photon_channels):

        _photon_timings += np.random.normal(self.pmt_transit_time_mean,
                                            self.pmt_transit_time_spread / 2.35482,
                                            len(_photon_timings)).astype(np.int64)
        
        #Why is this done here and additionally in the get_n_photons function of S1PhotonHits??
        _photon_is_dpe = np.random.binomial(n=1,
                                            p=self.p_double_pe_emision,
                                            size=len(_photon_timings)).astype(np.bool_)


        _photon_gains = self.gains[_photon_channels] \
            * loop_uniform_to_pe_arr(np.random.random(len(_photon_channels)), _photon_channels, self.__uniform_to_pe_arr)

        # Add some double photoelectron emission by adding another sampled gain
        n_double_pe = _photon_is_dpe.sum()
        _photon_gains[_photon_is_dpe] += self.gains[_photon_channels[_photon_is_dpe]] \
            * loop_uniform_to_pe_arr(np.random.random(n_double_pe), _photon_channels[_photon_is_dpe], self.__uniform_to_pe_arr) 

        return _photon_timings, _photon_gains, _photon_is_dpe
    
    def build_output(self, _photon_timings, _photon_channels, _photon_gains, _photon_is_dpe):

        result = np.zeros(_photon_channels.shape[0], dtype = self.dtype)
        result["time"] = _photon_timings
        result["channel"] = _photon_channels
        result["endtime"] = result["time"]
        result["photon_gain"] = _photon_gains 
        result["dpe"] = _photon_is_dpe

        return result
    
    def init_spe_scaling_factor_distributions(self):
        #This code will be duplicate with the corresponding S2 class 
        # Improve!!
        config="PLACEHOLDER_FIX_THIS_PART"
        h = deterministic_hash(config) # What is this part doing?
        #if h in self._cached_uniform_to_pe_arr:
        #    __uniform_to_pe_arr = self._cached_uniform_to_pe_arr[h]
        #    return __uniform_to_pe_arr

        # Extract the spe pdf from a csv file into a pandas dataframe
        spe_shapes = self.photon_area_distribution

        # Create a converter array from uniform random numbers to SPE gains (one interpolator per channel)
        # Scale the distributions so that they have an SPE mean of 1 and then calculate the cdf
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'].values  # / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            grid_cdf = np.linspace(0, 1, 2001)
            grid_scale = interp1d(cdf, scaled_bins,
                                  kind='next',
                                  bounds_error=False,
                                  fill_value=(scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)

        if len(uniform_to_pe_arr):
            __uniform_to_pe_arr = np.stack(uniform_to_pe_arr)
            self._cached_uniform_to_pe_arr[h] = __uniform_to_pe_arr

        log.debug('Spe scaling factors created, cached with key %s' % h)
        return __uniform_to_pe_arr
        
    
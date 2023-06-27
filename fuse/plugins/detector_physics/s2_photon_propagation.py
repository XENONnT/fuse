import strax
import straxen
import numpy as np
import logging

from numba import njit
from scipy.stats import skewnorm
from scipy.interpolate import interp1d

from strax import deterministic_hash
export, __all__ = strax.exporter()

from ...common import DummyMap, loop_uniform_to_pe_arr

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.S2_Signal')


@export
class S2PhotonPropagation(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("s2_photons", "extracted_electrons", "drifted_electrons", "s2_photons_sum")
    provides = "propagated_s2_photons"
    data_kind = "S2_photons"

    #Forbid rechunking
    rechunk_on_save = False

    dtype = [('channel', np.int64),
             ('dpe', np.bool_),
             ('photon_gain', np.int64),
            ]
    dtype = dtype + strax.time_fields

    #Config options
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

    triplet_lifetime_gas = straxen.URLConfig(
        type=(int, float),
        help='triplet_lifetime_gas',
    )

    singlet_lifetime_gas = straxen.URLConfig(
        type=(int, float),
        help='singlet_lifetime_gas',
    )

    triplet_lifetime_liquid = straxen.URLConfig(
        type=(int, float),
        help='triplet_lifetime_liquid',
    )

    singlet_lifetime_liquid = straxen.URLConfig(
        type=(int, float),
        help='singlet_lifetime_liquid',
    )

    s2_time_spread = straxen.URLConfig(
        type=(int, float),
        help='s2_time_spread',
    )

    s2_time_model = straxen.URLConfig(
        help='s2_time_model',
    )

    singlet_fraction_gas = straxen.URLConfig(
        type=(int, float),
        help='singlet_fraction_gas',
    )
    #needed as config?
    phase_s2 = straxen.URLConfig(
        default="gas",
        help='phase_s2',
    )

    s2_luminescence_model = straxen.URLConfig(
        help='s2_luminescence_model',
    )

    drift_velocity_liquid = straxen.URLConfig(
        type=(int, float),
        help='drift_velocity_liquid',
    )

    tpc_length = straxen.URLConfig(
        type=(int, float),
        help='tpc_length',
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

    tpc_radius = straxen.URLConfig(
        type=(int, float),
        help='tpc_radius',
    )

    n_top_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs on top array',
    )

    n_tpc_pmts = straxen.URLConfig(
        type=(int),
        help='Number of PMTs in the TPC',
    )

    diffusion_constant_transverse = straxen.URLConfig(
        type=(int, float),
        help='diffusion_constant_transverse',
    )

    s2_aft_skewness = straxen.URLConfig(
        type=(int, float),
        help='s2_aft_skewness',
    )

    s2_aft_sigma = straxen.URLConfig(
        type=(int, float),
        help='s2_aft_sigma',
    )
    
    enable_field_dependencies = straxen.URLConfig(
        help='enable_field_dependencies',
    )
    
    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    s2_pattern_map = straxen.URLConfig(
        cache=True,
        help='s2_pattern_map',
    )
    
    photon_area_distribution = straxen.URLConfig(
        cache=True,
        help='photon_area_distribution',
    )
    
    s2_optical_propagation_spline = straxen.URLConfig(
        cache=True,
        help='s2_optical_propagation_spline',
    )
    
    s2_luminescence_map = straxen.URLConfig(
        cache=True,
        help='s2_luminescence_map',
    )
    
    garfield_gas_gap_map = straxen.URLConfig(
        cache=True,
        help='garfield_gas_gap_map',
    )
    #stupid naming problem...
    field_dependencies_map_tmp = straxen.URLConfig(
        help='field_dependencies_map',
    )

    deterministic_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running S2PhotonPropagation in debug mode")
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

        #Set the random generator for scipy
        skewnorm.random_state=self.rng
        
        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]
        
        #Move this part into a nice URLConfig protocol?
        # Field dependencies 
        if any(self.enable_field_dependencies.values()):
            self.drift_velocity_scaling = 1.0
            # calculating drift velocity scaling to match total drift time for R=0 between cathode and gate
            if "norm_drift_velocity" in self.enable_field_dependencies.keys():
                if self.enable_field_dependencies['norm_drift_velocity']:
                    norm_dvel = self.field_dependencies_map_tmp(np.array([ [0], [- self.tpc_length]]).T, map_name='drift_speed_map')[0]
                    norm_dvel*=1e-4
                    drift_velocity_scaling = self.drift_velocity_liquid/norm_dvel
            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                return self.field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)
            self.field_dependencies_map = rz_map

            
        self._cached_uniform_to_pe_arr = {}
        self.__uniform_to_pe_arr = self.init_spe_scaling_factor_distributions()


    def compute(self, individual_electrons, interactions_in_roi):

        #Just apply this to clusters with photons
        mask = interactions_in_roi["n_electron_extracted"] > 0

        if len(individual_electrons) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        positions = np.array([interactions_in_roi[mask]["x"], interactions_in_roi[mask]["y"]]).T
        
        _photon_channels = self.photon_channels(interactions_in_roi[mask]["n_electron_extracted"],
                                                interactions_in_roi[mask]["z_obs"],
                                                positions,
                                                interactions_in_roi[mask]["drift_time_mean"] ,
                                                interactions_in_roi[mask]["sum_s2_photons"],
                                               )
        #_photon_channels = _photon_channels.astype(np.int64)
        _photon_timings = self.photon_timings(positions,
                                              interactions_in_roi[mask]["sum_s2_photons"],
                                              _photon_channels,
                                             )
        
        #repeat for n photons per electron # Should this be before adding delays?
        _photon_timings += np.repeat(individual_electrons["time"], individual_electrons["n_s2_photons"])
        
        #Do i want to save both -> timings with and without pmt transition time spread?
        # Correct for PMT Transition Time Spread (skip for pmt after-pulses)
        # note that PMT datasheet provides FWHM TTS, so sigma = TTS/(2*sqrt(2*log(2)))=TTS/2.35482
        _photon_timings += self.rng.normal(self.pmt_transit_time_mean,
                                            self.pmt_transit_time_spread / 2.35482,
                                            len(_photon_timings)).astype(np.int64)
        _photon_is_dpe = self.rng.binomial(n=1,
                                            p=self.p_double_pe_emision,
                                            size=len(_photon_timings)).astype(np.bool_)
        _photon_gains = self.gains[_photon_channels] \
            * loop_uniform_to_pe_arr(self.rng.random(len(_photon_channels)), _photon_channels, self.__uniform_to_pe_arr)

        # Add some double photoelectron emission by adding another sampled gain
        n_double_pe = _photon_is_dpe.sum()
        _photon_gains[_photon_is_dpe] += self.gains[_photon_channels[_photon_is_dpe]] \
            * loop_uniform_to_pe_arr(self.rng.random(n_double_pe), _photon_channels[_photon_is_dpe], self.__uniform_to_pe_arr) 

        
        #we may need to sort this output by time and channel??
        result = np.zeros(len(_photon_channels), dtype = self.dtype)
        result["channel"] = _photon_channels
        result["time"] = _photon_timings
        result["endtime"] = _photon_timings
        result["dpe"] = _photon_is_dpe
        result["photon_gain"] = _photon_gains
        
        return result
    
       
    def photon_channels(self, n_electron, z_obs, positions, drift_time_mean, n_photons):
        
        channels = np.arange(self.n_tpc_pmts).astype(np.int64)
        top_index = np.arange(self.n_top_pmts)
        channels_bottom = np.arange(self.n_top_pmts, self.n_tpc_pmts)
        bottom_index = np.array(channels_bottom)
        
        if self.diffusion_constant_transverse > 0:
            pattern = self.s2_pattern_map_diffuse(n_electron, z_obs, positions, drift_time_mean)  # [position, pmt]
        else:
            pattern = self.s2_pattern_map(positions)  # [position, pmt]
            
        if pattern.shape[1] - 1 not in bottom_index:
            pattern = np.pad(pattern, [[0, 0], [0, len(bottom_index)]], 
                             'constant', constant_values=1)
        
        # Remove turned off pmts
        pattern[:, np.in1d(channels, self.turned_off_pmts)] = 0
        
        sum_pat = np.sum(pattern, axis=1).reshape(-1, 1)
        pattern = np.divide(pattern, sum_pat, out=np.zeros_like(pattern), where=sum_pat != 0)

        assert pattern.shape[0] == len(positions)
        assert pattern.shape[1] == len(channels)
        
        _buffer_photon_channels = []
        # Randomly assign to channel given probability of each channel
        for i, n_ph in enumerate(n_photons):
            pat = pattern[i]
            
            #Redistribute pattern with user specified aft smearing
            if self.s2_aft_sigma != 0: 
                _cur_aft=np.sum(pat[top_index])/np.sum(pat)
                _new_aft=_cur_aft*skewnorm.rvs(loc=1.0, scale=self.s2_aft_sigma, a=self.s2_aft_skewness)
                _new_aft=np.clip(_new_aft, 0, 1)
                pat[top_index]*=(_new_aft/_cur_aft)
                pat[bottom_index]*=(1 - _new_aft)/(1 - _cur_aft)
            
            # Pattern map return zeros   
            if np.isnan(pat).sum() > 0:  
                _photon_channels = np.array([-1] * n_ph)
                
            else:
                _photon_channels = self.rng.choice(channels,
                                                    size=n_ph,
                                                    p=pat,
                                                    replace=True)
                
            _buffer_photon_channels.append(_photon_channels)
        
        _photon_channels = np.concatenate(_buffer_photon_channels)
        
        return _photon_channels.astype(np.int64)

    
    def s2_pattern_map_diffuse(self, n_electron, z, xy, drift_time_mean):
        """Returns an array of pattern of shape [n interaction, n PMTs]
        pattern of each interaction is an average of n_electron patterns evaluated at
        diffused position near xy. The diffused positions sample from 2d symmetric gaussian
            with spread scale with sqrt of drift time.
        :param n_electron: a 1d int array
        :param z: a 1d float array
        :param xy: a 2d float array of shape [n interaction, 2]
        :param config: dict of the wfsim config
        :param resource: instance of the resource class
        """
        assert all(z < 0), 'All S2 in liquid should have z < 0'

        if self.enable_field_dependencies['diffusion_transverse_map']:
            diffusion_constant_radial = self.field_dependencies_map(z, xy, map_name='diffusion_radial_map')  # cm²/s
            diffusion_constant_azimuthal = self.field_dependencies_map(z, xy, map_name='diffusion_azimuthal_map') # cm²/s
            diffusion_constant_radial *= 1e-9  # cm²/ns
            diffusion_constant_azimuthal *= 1e-9  # cm²/ns
        else:
            #diffusion_constant_transverse = diffusion_constant_transverse
            diffusion_constant_radial = self.diffusion_constant_transverse
            diffusion_constant_azimuthal = self.diffusion_constant_transverse

        hdiff_stdev_radial = np.sqrt(2 * diffusion_constant_radial * drift_time_mean)
        hdiff_stdev_azimuthal = np.sqrt(2 * diffusion_constant_azimuthal * drift_time_mean)

        hdiff_radial = self.rng.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_radial, n_electron, axis=0)
        hdiff_azimuthal = self.rng.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_azimuthal, n_electron, axis=0)
        hdiff = np.column_stack([hdiff_radial, hdiff_azimuthal])
        theta = np.arctan2(xy[:,1], xy[:,0])
        matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
        hdiff = np.vstack([(matrix[i] @ np.split(hdiff, np.cumsum(n_electron))[:-1][i].T).T for i in range(len(matrix))])
        # Should we also output this xy position in truth?
        xy_multi = np.repeat(xy, n_electron, axis=0) + hdiff  # One entry xy per electron
        # Remove points outside tpc, and the pattern will be the average inside tpc
        # Should be done naturally with the s2 pattern map, however, there's some bug there, so we apply this hard cut
        mask = np.sum(xy_multi ** 2, axis=1) <= self.tpc_radius ** 2

        if isinstance(self.s2_pattern_map, DummyMap):
            output_dim = self.s2_pattern_map.shape[-1]
        else:
            output_dim = self.s2_pattern_map.data['map'].shape[-1]
        pattern = np.zeros((len(n_electron), output_dim))
        n0 = 0
        # Average over electrons for each s2
        for ix, ne in enumerate(n_electron):
            s = slice(n0, n0+ne)
            pattern[ix, :] = np.average(self.s2_pattern_map(xy_multi[s][mask[s]]), axis=0)
            n0 += ne

        return pattern
    
    
    def photon_timings(self, positions, n_photons, _photon_channels):
        
        if self.s2_luminescence_model == "simple":
            print("Not Implemented!")
            #_photon_timings = luminescence_timings_simple(positions, n_photons_per_xy)
        elif self.s2_luminescence_model == "garfield":
            print("Not Implemented!")
        elif self.s2_luminescence_model == "garfield_gas_gap":
            _photon_timings = self.luminescence_timings_garfield_gasgap(positions, n_photons)
        else:
            raise KeyError(f"{self.s2_luminescence_model} is not valid! Use 'simple' or 'garfield' or 'garfield_gas_gap'")
            
        # Emission Delay
        _photon_timings += self.singlet_triplet_delays(len(_photon_timings), self.singlet_fraction_gas)
        
        # Optical Propagation Delay
        if "optical_propagation" in self.s2_time_model:
            # optical propagation splitting top and bottom
            _photon_timings += self.optical_propagation(_photon_channels)
        elif "zero_delay" in self.s2_time_model:
            # no optical propagation delay
            _photon_timings += np.zeros_like(_photon_timings, dtype=np.int64)
        elif "s2_time_spread around zero" in self.s2_time_model:
            # simple/existing delay
            _photon_timings += self.rng.normal(0, self.s2_time_spread, len(_photon_timings)).astype(np.int64)
        else:
            raise KeyError(f"{self.s2_time_model} is not in any of the valid s2 time models")
        
        return _photon_timings

    def singlet_triplet_delays(self, size, singlet_ratio):
        """
        Given the amount of the excimer, return time between excimer decay
        and their time of generation.
        size           - amount of excimer
        self.phase     - 'liquid' or 'gas'
        singlet_ratio  - fraction of excimers that become singlets
                         (NOT the ratio of singlets/triplets!)
        """
        if self.phase_s2 == 'liquid':
            t1, t3 = (self.singlet_lifetime_liquid,
                      self.triplet_lifetime_liquid)
        elif self.phase_s2 == 'gas':
            t1, t3 = (self.singlet_lifetime_gas,
                      self.triplet_lifetime_gas)
        else:
            t1, t3 = 0, 0

        delay = self.rng.choice([t1, t3], size, replace=True,
                                 p=[singlet_ratio, 1 - singlet_ratio])
        return (self.rng.exponential(1, size) * delay).astype(np.int64)

    
    def optical_propagation(self, channels):
        """Function getting times from s2 timing splines:
        :param channels: The channels of all s2 photon
        """
        prop_time = np.zeros_like(channels)
        u_rand = self.rng.random(len(channels))[:, None]

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s2_optical_propagation_spline(u_rand[is_top], map_name='top')

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s2_optical_propagation_spline(u_rand[is_bottom], map_name='bottom')

        return prop_time.astype(np.int64)
    
    
    def luminescence_timings_garfield_gasgap(self, xy, n_photons):
        """
        Luminescence time distribution computation according to garfield scintillation maps
        which are ONLY drawn from below the anode, and at different gas gaps
        :param xy: 1d array with positions
        :param n_photons: 1d array with ints for number of xy positions
        returns 2d array with ints for photon timings of input param 'shape'
        """
        #assert 's2_luminescence_gg' in resource.__dict__, 's2_luminescence_gg model not found'
        assert len(n_photons) == len(xy), 'Input number of n_electron should have same length as positions'

        d_gasgap = self.s2_luminescence_map['gas_gap'][1]-self.s2_luminescence_map['gas_gap'][0]

        cont_gas_gaps = self.garfield_gas_gap_map(xy)
        draw_index = np.digitize(cont_gas_gaps, self.s2_luminescence_map['gas_gap'])-1
        diff_nearest_gg = cont_gas_gaps - self.s2_luminescence_map['gas_gap'][draw_index]

        return draw_excitation_times(self.s2_luminescence_map['timing_inv_cdf'],
                                     draw_index,
                                     n_photons,
                                     diff_nearest_gg,
                                     d_gasgap,
                                     self.rng
                                    )
    
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

@njit
def draw_excitation_times(inv_cdf_list, hist_indices, nph, diff_nearest_gg, d_gas_gap, rng):
    
    """
    Draws the excitation times from the GARFIELD electroluminescence map
    
    :param inv_cdf_list:       List of inverse CDFs for the excitation time histograms
    :param hist_indices:       The index of the histogram which refers to the gas gap
    :param nph:                A 1-d array of the number of photons per electron
    :param diff_nearest_gg:    The difference between the gas gap from the map
                               (continuous value) and the nearest (discrete) value
                               of the gas gap corresponding to the excitation time
                               histograms
    :param d_gas_gap:          Spacing between two consecutive gas gap values
    
    returns time of each photon
    """
    
    inv_cdf_len = len(inv_cdf_list[0])
    timings = np.zeros(np.sum(nph))
    upper_hist_ind = np.clip(hist_indices+1, 0, len(inv_cdf_list)-1)
    
    count = 0
    for i, (hist_ind, u_hist_ind, n, dngg) in enumerate(zip(hist_indices,
                                                            upper_hist_ind, 
                                                            nph,
                                                            diff_nearest_gg)):
        
        #There are only 10 values of gas gap separated by 0.1mm, so we interpolate
        #between two histograms
        
        interp_cdf = ((inv_cdf_list[u_hist_ind]-inv_cdf_list[hist_ind])*(dngg/d_gas_gap)
                       +inv_cdf_list[hist_ind])
        
        #Subtract 2 because this way we don't want to sample from this last strange tail
        samples = rng.uniform(0, inv_cdf_len-2, n)
        #samples = np.random.uniform(0, inv_cdf_len-2, n)
        t1 = interp_cdf[np.floor(samples).astype('int')]
        t2 = interp_cdf[np.ceil(samples).astype('int')]
        T = (t2-t1)*(samples - np.floor(samples))+t1
        if n!=0:
            T = T-np.mean(T)
        
        #subtract mean to get proper drift time and z correlation
        timings[count:count+n] = T
        count+=n
    return timings
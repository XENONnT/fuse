import strax
import straxen
import numpy as np
import os

from numba import njit
from scipy.stats import skewnorm
from scipy.interpolate import interp1d

from ..common import make_map

import wfsim

from strax import deterministic_hash


private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')

@strax.takes_config(
    strax.Option('s2_aft_sigma', default=config["s2_aft_sigma"], track=False, infer_type=False,
                 help="s2_aft_sigma"),
    strax.Option('s2_aft_skewness', default=config["s2_aft_skewness"], track=False, infer_type=False,
                 help="s2_aft_skewness"),
    strax.Option('diffusion_constant_transverse', default=config["diffusion_constant_transverse"], track=False, infer_type=False,
                 help="diffusion_constant_transverse"),
    strax.Option('n_top_pmts', default=253, track=False, infer_type=False,
                 help="n_top_pmts"),
    strax.Option('n_tpc_pmts', default=494, track=False, infer_type=False,
                 help="n_tpc_pmts"),
    strax.Option('tpc_radius', default=config["tpc_radius"], track=False, infer_type=False,
                 help="tpc_radius"),
    strax.Option('to_pe_file', default=os.path.join(private_files_path,"sim_files/to_pe_nt.npy"), track=False, infer_type=False,
                 help="to_pe file"),
    strax.Option('digitizer_voltage_range', default=config['digitizer_voltage_range'], track=False, infer_type=False,
                 help="digitizer_voltage_range"),
    strax.Option('digitizer_bits', default=config['digitizer_bits'], track=False, infer_type=False,
                 help="digitizer_bits"),
    strax.Option('pmt_circuit_load_resistor', default=config['pmt_circuit_load_resistor'], track=False, infer_type=False,
                 help="pmt_circuit_load_resistor"),
    strax.Option('s2_pattern_map_file',
                 default=os.path.join(private_files_path,"sim_files/XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl"),
                 track=False,
                 infer_type=False,
                 help="s2_pattern_map"),
    strax.Option('field_dependencies_map',
                 default=os.path.join(private_files_path,"sim_files/field_dependent_radius_depth_maps_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"),
                 track=False,
                 infer_type=False,
                 help="field_dependencies_map"),
    strax.Option('tpc_length', default=config['tpc_length'], track=False, infer_type=False,
                 help="tpc_length"),
    strax.Option('drift_velocity_liquid', default=config['drift_velocity_liquid'], track=False, infer_type=False,
                 help="drift_velocity_liquid"),
    strax.Option('s2_luminescence_model', default=config['s2_luminescence_model'], track=False, infer_type=False,
                 help="s2_luminescence_model"),
    strax.Option('phase_s2', default="gas", track=False, infer_type=False,
                 help="phase"),
    strax.Option('singlet_fraction_gas', default=config['singlet_fraction_gas'], track=False, infer_type=False,
                 help="singlet_fraction_gas"),
    strax.Option('s2_time_model', default=config['s2_time_model'], track=False, infer_type=False,
                 help="s2_time_model"),
    strax.Option('s2_time_spline', default=config['s2_time_spline'], track=False, infer_type=False,
                 help="s2_time_spline"),
    strax.Option('s2_time_spread', default=config['s2_time_spread'], track=False, infer_type=False,
                 help="s2_time_spread"),
    strax.Option('singlet_lifetime_liquid', default=config['singlet_lifetime_liquid'], track=False, infer_type=False,
                 help="singlet_lifetime_liquid"),
    strax.Option('triplet_lifetime_liquid', default=config['triplet_lifetime_liquid'], track=False, infer_type=False,
                 help="triplet_lifetime_liquid"),
    strax.Option('singlet_lifetime_gas', default=config['singlet_lifetime_gas'], track=False, infer_type=False,
                 help="singlet_lifetime_gas"),
    strax.Option('triplet_lifetime_gas', default=config['triplet_lifetime_gas'], track=False, infer_type=False,
                 help="triplet_lifetime_gas"),
    strax.Option('pmt_transit_time_mean', default=config['pmt_transit_time_mean'], track=False, infer_type=False,
                 help="pmt_transit_time_mean"),
    strax.Option('pmt_transit_time_spread', default=config['pmt_transit_time_spread'], track=False, infer_type=False,
                 help="pmt_transit_time_spread"),
    strax.Option('p_double_pe_emision', default=config['p_double_pe_emision'], track=False, infer_type=False,
                 help="p_double_pe_emision"),
    strax.Option('photon_area_distribution', default=config['photon_area_distribution'], track=False, infer_type=False,
                 help="photon_area_distribution"),
)
class S2_photon_distributions_and_timing(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("photons", "extracted_electrons", "drifted_electrons", "sum_photons")
    provides = "photon_channels_and_timeing"
    data_kind = "S2_photons"

    #Forbid rechunking
    rechunk_on_save = False

    dtype = [('channel', np.int64),
             ('dpe', np.bool_),
             ('photon_gain', np.int64),
            ]
    dtype = dtype + strax.time_fields
    
    def setup(self):
        
        to_pe = straxen.get_resource(self.to_pe_file, fmt='npy')
        self.to_pe = to_pe[0][1]

        adc_2_current = (self.digitizer_voltage_range
                / 2 ** (self.digitizer_bits)
                 / self.pmt_circuit_load_resistor)

        self.gains = np.divide(adc_2_current,
                          self.to_pe,
                          out=np.zeros_like(self.to_pe),
                          where=self.to_pe != 0)

        self.pmt_mask = np.array(self.gains) > 0  # Converted from to pe (from cmt by default)
        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]

        self.s2_pattern_map = wfsim.make_patternmap(self.s2_pattern_map_file, fmt='pkl', pmt_mask=self.pmt_mask)
        
        
        # Field dependencies 
        # This config entry a dictionary of 5 items
        self.enable_field_dependencies = config['enable_field_dependencies'] #This is not so nice
        if any(self.enable_field_dependencies.values()):
            field_dependencies_map_tmp = make_map(self.field_dependencies_map, fmt='json.gz', method='RectBivariateSpline')
            self.drift_velocity_scaling = 1.0
            # calculating drift velocity scaling to match total drift time for R=0 between cathode and gate
            if "norm_drift_velocity" in self.enable_field_dependencies.keys():
                if self.enable_field_dependencies['norm_drift_velocity']:
                    norm_dvel = field_dependencies_map_tmp(np.array([ [0], [- self.tpc_length]]).T, map_name='drift_speed_map')[0]
                    norm_dvel*=1e-4
                    drift_velocity_scaling = self.drift_velocity_liquid/norm_dvel
            def rz_map(z, xy, **kwargs):
                r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                return field_dependencies_map_tmp(np.array([r, z]).T, **kwargs)
            self.field_dependencies_map = rz_map
        
        if 'garfield_gas_gap' in self.s2_luminescence_model:
            #garfield_gas_gap option is using (x,y) -> gas gap (from the map) -> s2 luminescence
            #from garfield. This s2_luminescence_gg is indexed only by the gas gap, and
            #corresponds to electrons drawn directly below the anode
            self.s2_luminescence_map = straxen.get_resource(os.path.join(private_files_path,"sim_files/garfield_timing_map_gas_gap_sr0.npy"), fmt='npy')
            self.garfield_gas_gap_map = make_map(os.path.join(private_files_path,"sim_files/garfield_gas_gap_map_sr0.json"), fmt = 'json')
        
        if self.s2_time_spline:
            self.s2_optical_propagation_spline = make_map(os.path.join(private_files_path,"sim_files/XENONnT_s2_opticalprop_time_v0.json.gz"), fmt="json.gz")

        self.photon_area_distribution = straxen.get_resource(self.photon_area_distribution, fmt='csv')
        self._cached_uniform_to_pe_arr = {}
        self.__uniform_to_pe_arr = self.init_spe_scaling_factor_distributions()


    def compute(self, individual_electrons, electron_cloud):

        if len(individual_electrons) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        positions = np.array([electron_cloud["x"], electron_cloud["y"]]).T
        
        _photon_channels = self.photon_channels(electron_cloud["n_electron_extracted"],
                                                electron_cloud["z_obs"],
                                                positions,
                                                electron_cloud["drift_time_mean"] ,
                                                electron_cloud["sum_photons"],
                                               )
        #_photon_channels = _photon_channels.astype(np.int64)
        _photon_timings = self.photon_timings(positions,
                                              electron_cloud["sum_photons"],
                                              _photon_channels,
                                             )
        
        #repeat for n photons per electron # Should this be before adding delays?
        _photon_timings += np.repeat(individual_electrons["time"], individual_electrons["n_photons"])
        
        #Do i want to save both -> timings with and without pmt transition time spread?
        # Correct for PMT Transition Time Spread (skip for pmt after-pulses)
        # note that PMT datasheet provides FWHM TTS, so sigma = TTS/(2*sqrt(2*log(2)))=TTS/2.35482
        _photon_timings += np.random.normal(self.pmt_transit_time_mean,
                                            self.pmt_transit_time_spread / 2.35482,
                                            len(_photon_timings)).astype(np.int64)
        _photon_is_dpe = np.random.binomial(n=1,
                                            p=self.p_double_pe_emision,
                                            size=len(_photon_timings)).astype(np.bool_)
        _photon_gains = self.gains[_photon_channels] \
            * loop_uniform_to_pe_arr(np.random.random(len(_photon_channels)), _photon_channels, self.__uniform_to_pe_arr)

        # Add some double photoelectron emission by adding another sampled gain
        n_double_pe = _photon_is_dpe.sum()
        _photon_gains[_photon_is_dpe] += self.gains[_photon_channels[_photon_is_dpe]] \
            * loop_uniform_to_pe_arr(np.random.random(n_double_pe), _photon_channels[_photon_is_dpe], self.__uniform_to_pe_arr) 

        
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
                _photon_channels = np.random.choice(channels,
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

        hdiff_radial = np.random.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_radial, n_electron, axis=0)
        hdiff_azimuthal = np.random.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_azimuthal, n_electron, axis=0)
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
            _photon_timings += np.random.normal(0, self.s2_time_spread, len(_photon_timings)).astype(np.int64)
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

        delay = np.random.choice([t1, t3], size, replace=True,
                                 p=[singlet_ratio, 1 - singlet_ratio])
        return (np.random.exponential(1, size) * delay).astype(np.int64)

    
    def optical_propagation(self, channels):
        """Function getting times from s2 timing splines:
        :param channels: The channels of all s2 photon
        """
        prop_time = np.zeros_like(channels)
        u_rand = np.random.rand(len(channels))[:, None]

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
                                     d_gasgap
                                    )
    
    def init_spe_scaling_factor_distributions(self):
        #This code will be duplicate with the corresponding S2 class 
        # Improve!!
        
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

        return __uniform_to_pe_arr
        #log.debug('Spe scaling factors created, cached with key %s' % h)
    
    
@njit
def draw_excitation_times(inv_cdf_list, hist_indices, nph, diff_nearest_gg, d_gas_gap):
    
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
        samples = np.random.uniform(0, inv_cdf_len-2, n)
        t1 = interp_cdf[np.floor(samples).astype('int')]
        t2 = interp_cdf[np.ceil(samples).astype('int')]
        T = (t2-t1)*(samples - np.floor(samples))+t1
        if n!=0:
            T = T-np.mean(T)
        
        #subtract mean to get proper drift time and z correlation
        timings[count:count+n] = T
        count+=n
    return timings

#This is a modified version of the corresponding WFsim code....
@njit()
def uniform_to_pe_arr(p, channel, __uniform_to_pe_arr):
    indices = np.int64(p * 2000) + 1
    return __uniform_to_pe_arr[channel, indices]

#In WFSim uniform_to_pe_arr is called inside a loop over the channels
#I needed to change the code to run on all channels at once
@njit()
def loop_uniform_to_pe_arr(p, channel, __uniform_to_pe_arr):
    result = []
    for i in range(len(p)):
        result.append(uniform_to_pe_arr(p[i],
                                        channel=channel[i],
                                        __uniform_to_pe_arr=__uniform_to_pe_arr) )
    return np.array(result)
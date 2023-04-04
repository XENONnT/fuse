import numpy as np
import strax
import straxen
import wfsim
import logging 
import nestpy
import os

from wfsim.load_resource import DummyMap

private_files_path = "path/to/private/files"
config = straxen.get_resource(os.path.join(private_files_path, 'sim_files/fax_config_nt_sr0_v4.json') , fmt='json')

@strax.takes_config(
    strax.Option('s1_detection_efficiency', default=1, track=False, infer_type=False,
                 help="Some placeholder for s1_detection_efficiency"),
    strax.Option('p_double_pe_emision', default=config["p_double_pe_emision"], track=False, infer_type=False,
                 help="Some placeholder for p_double_pe_emision"),
    strax.Option('s1_lce_correction_map',
                 default=os.path.join(private_files_path, "sim_files/XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz"),
                 track=False,
                 infer_type=False,
                 help="S1 LCE correction map"),
    strax.Option('s1_pattern_map',
                 default=os.path.join(private_files_path, "sim_files/XENONnT_s1_xyz_patterns_corrected_qes_MCva43fa9b_wires.pkl"),
                 track=False,
                 infer_type=False,
                 help="S1 pattern map"),
    strax.Option('n_tpc_pmts', default=494, track=False, infer_type=False,
                 help="Number of PMTs in the TPC"),
    strax.Option('n_top_pmts', default=253, track=False, infer_type=False,
                 help="Number of PMTs on top array"),
    strax.Option('digitizer_voltage_range', default=config['digitizer_voltage_range'], track=False, infer_type=False,
                 help="digitizer_voltage_range"),
    strax.Option('digitizer_bits', default=config['digitizer_bits'], track=False, infer_type=False,
                 help="digitizer_bits"),
    strax.Option('pmt_circuit_load_resistor', default=config['pmt_circuit_load_resistor'], track=False, infer_type=False,
                 help="pmt_circuit_load_resistor"),
    strax.Option('to_pe_file', default=os.path.join(private_files_path, "sim_files/to_pe_nt.npy"), track=False, infer_type=False,
                 help="to_pe file"),
    strax.Option('s1_model_type', default=config['s1_model_type'], track=True, infer_type=False,
                 help="s1_model_type"),
    strax.Option('s1_time_spline',
                 default=os.path.join(private_files_path, "sim_files/XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz"),
                 track=False,
                 infer_type=False,
                 help="S1 Time Spline"),
    strax.Option('s1_decay_time', default=config['s1_decay_time'], track=False, infer_type=False,
                 help="s1_decay_time"),
    strax.Option('s1_decay_spread', default=config['s1_decay_spread'], track=False, infer_type=False,
                 help="s1_decay_spread"),
    strax.Option('phase', default="liquid", track=False, infer_type=False,
                 help="xenon phase"),
    strax.Option('maximum_recombination_time', default=config["maximum_recombination_time"], track=False, infer_type=False,
                 help="maximum_recombination_time"),
)
class S1_scintillation_and_propagation(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("wfsim_instructions")
    provides = "S1_channel_and_timings"
    
    dtype = [('photon_channel', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def setup(self):
        
        self.s1_lce_correction_map = make_map(self.s1_lce_correction_map, fmt='json.gz')
        self.s1_pattern_map = wfsim.make_patternmap(self.s1_pattern_map, fmt='pkl', pmt_mask=None)
        
        to_pe = straxen.get_resource(self.to_pe_file, fmt='npy')
        self.to_pe = to_pe[0][1]
        
        adc_2_current = (self.digitizer_voltage_range
                / 2 ** (self.digitizer_bits)
                 / self.pmt_circuit_load_resistor)

        gains = np.divide(adc_2_current,
                          self.to_pe,
                          out=np.zeros_like(self.to_pe),
                          where=self.to_pe != 0)

        self.turned_off_pmts = np.arange(len(gains))[np.array(gains) == 0]
        
        
        self.s1_optical_propagation_spline = make_map(self.s1_time_spline,
                                                      fmt='json.gz',
                                                      method='RegularGridInterpolator')
        
        if 'nest' in self.s1_model_type: #and (self.nestpy_calc is None):
            #log.info('Using NEST for scintillation time without set calculator\n'
            #         'Creating new nestpy calculator')
            self.nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())


    def compute(self, wfsim_instructions):
        
        #And do this part only for S1 signals
        instruction = wfsim_instructions[wfsim_instructions["type"] == 1]
        
        t = instruction['time']
        x = instruction['x']
        y = instruction['y']
        z = instruction['z']
        n_photons = instruction['amp']
        recoil_type = instruction['recoil']
        positions = np.array([x, y, z]).T  # For map interpolation
        
        n_photon_hits = self.get_n_photons(n_photons=n_photons,
                                           positions=positions,
                                           )
        
        # The new way interpolation is written always require a list
        _photon_channels = self.photon_channels(positions=positions,
                                                n_photon_hits=n_photon_hits,
                                                )
        
        extra_targs = {}
        if 'nest' in self.s1_model_type:
            extra_targs['n_photons_emitted'] = n_photons
            extra_targs['n_excitons'] = instruction['n_excitons']
            extra_targs['local_field'] = instruction['local_field']
            extra_targs['e_dep'] = instruction['e_dep']
            extra_targs['nestpy_calc'] = self.nestpy_calc
            
        _photon_timings = self.photon_timings(t=t,
                                              n_photon_hits=n_photon_hits, 
                                              recoil_type=recoil_type,
                                              channels=_photon_channels,
                                              positions=positions,
                                              **extra_targs
                                             )
        
        sortind = np.argsort(_photon_channels)

        _photon_channels = _photon_channels[sortind]
        _photon_timings = _photon_timings[sortind]
        
        result = np.zeros(_photon_channels.shape[0], dtype = self.dtype)
        result["photon_channel"] = _photon_channels
        result["time"] = _photon_timings
        result["endtime"] = result["time"]
        
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
    
    
    def photon_channels(self, positions, n_photon_hits):
        """Calculate photon arrival channels
        :params positions: 2d array with xy positions of interactions
        :params n_photon_hits: 1d array of ints with number of photon hits to simulate
        :params config: dict wfsim config
        :params s1_pattern_map: interpolator instance of the s1 pattern map
        returns nested array with photon channels   
        """
        channels = np.arange(self.n_tpc_pmts)  # +1 for the channel map
        p_per_channel = self.s1_pattern_map(positions)
        p_per_channel[:, np.in1d(channels, self.turned_off_pmts)] = 0

        _photon_channels = np.array([]).astype(np.int64)
        for ppc, n in zip(p_per_channel, n_photon_hits):
            _photon_channels = np.append(_photon_channels,
                                         np.random.choice(
                                             channels,
                                             size=n,
                                             p=ppc / np.sum(ppc),
                                             replace=True))
        return _photon_channels
        
    def photon_timings(self,
                       t,
                       n_photon_hits,
                       recoil_type,
                       channels=None,
                       positions=None,
                       e_dep=None,
                       n_photons_emitted=None,
                       n_excitons=None, 
                       local_field=None,
                       nestpy_calc=None
                      ):
        """Calculate distribution of photon arrival timnigs
        :param t: 1d array of ints
        :param n_photon_hits: number of photon hits, 1d array of ints
        :param recoil_type: 1d array of ints
        :param config: dict wfsim config
        :param channels: list of photon hit channels 
        :param positions: nx3 array of true XYZ positions from instruction
        :param e_dep: energy of the deposit, 1d float array
        :param n_photons_emitted: number of orignally emitted photons/quanta, 1d int array
        :param n_excitons: number of exctions in deposit, 1d int array
        :param local_field: local field in the point of the deposit, 1d array of floats
        returns photon timing array"""
        _photon_timings = np.repeat(t, n_photon_hits)
        _n_hits_total = len(_photon_timings)

        if len(_photon_timings) == 0:
            return _photon_timings.astype(np.int64)

        if 'optical_propagation' in self.s1_model_type:
            z_positions = np.repeat(positions[:, 2], n_photon_hits)
            _photon_timings += self.optical_propagation(channels,
                                                        z_positions,
                                                        ).astype(np.int64)

        if 'simple' in self.s1_model_type:
            # Simple S1 model enabled: use it for ER and NR.
            _photon_timings += np.random.exponential(self.s1_decay_time, _n_hits_total).astype(np.int64)
            _photon_timings += np.random.normal(0, self.s1_decay_spread, _n_hits_total).astype(np.int64)

        if 'nest' in self.s1_model_type or 'custom' in self.s1_model_type:
            # Pulse model depends on recoil type
            counts_start = 0
            for i, counts in enumerate(n_photon_hits):

                if 'custom' in self.s1_model_type:
                    raise ValueError('Custom Model not implemented! ') 
         #           for k in vars(NestId):
         #               if k.startswith('_'):
         #                   continue
         #               if recoil_type[i] in getattr(NestId, k):
         #                   str_recoil_type = k
         #           try:
         #               _photon_timings[counts_start: counts_start + counts] += \
         #                   getattr(S1, str_recoil_type.lower())(
         #                   size=counts,
         #                   config=config,
         #                   phase=phase).astype(np.int64)
         #           except AttributeError:
         #               raise AttributeError(f"Recoil type must be ER, NR, alpha or LED, "
         #                                    f"not {recoil_type}. Check nest ids")

                if 'nest' in self.s1_model_type:
                    # Allow overwriting with "override_s1_photon_time_field"
                    # xenon:j_angevaare:wfsim_photon_timing_bug
                    #_local_field = config.get('override_s1_photon_time_field', local_field[i])
                    #_local_field = (_local_field if _local_field >0 else local_field[i])
                    _local_field = local_field[i]
                    scint_time = nestpy_calc.GetPhotonTimes(nestpy.INTERACTION_TYPE(recoil_type[i]),
                                                            n_photons_emitted[i],
                                                            n_excitons[i],
                                                            _local_field,
                                                            e_dep[i]
                                                           )

                    scint_time = np.clip(scint_time, 0, self.maximum_recombination_time)

                    # The first part of the scint_time is from exciton only, see
                    # https://github.com/NESTCollaboration/nestpy/blob/fe3d5d7da5d9b33ac56fbea519e02ef55152bc1d/src/nestpy/NEST.cpp#L164-L179
                    _photon_timings[counts_start: counts_start + counts] += \
                       np.random.choice(scint_time, counts, replace=False).astype(np.int64)

                counts_start += counts

        return _photon_timings
    
    def optical_propagation(self, channels, z_positions):
        """Function gettting times from s1 timing splines:
        :param channels: The channels of all s1 photon
        :param z_positions: The Z positions of all s1 photon
        :param config: current configuration of wfsim
        :param spline: pointer to s1 optical propagation splines from resources
        """
        assert len(z_positions) == len(channels), 'Give each photon a z position'

        prop_time = np.zeros_like(channels)
        z_rand = np.array([z_positions, np.random.rand(len(channels))]).T

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s1_optical_propagation_spline(z_rand[is_top], map_name='top')

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s1_optical_propagation_spline(z_rand[is_bottom], map_name='bottom')

        return prop_time
    


def make_map(map_file, fmt=None, method='WeightedNearestNeighbors'):
    """Fetch and make an instance of InterpolatingMap based on map_file
    Alternatively map_file can be a list of ["constant dummy", constant: int, shape: list]
    return an instance of  DummyMap"""

    if isinstance(map_file, list):
        assert map_file[0] == 'constant dummy', ('Alternative file input can only be '
                                                 '("constant dummy", constant: int, shape: list')
        return DummyMap(map_file[1], map_file[2])

    elif isinstance(map_file, str):
        if fmt is None:
            fmt = parse_extension(map_file)

        #log.debug(f'Initialize map interpolator for file {map_file}')
        map_data = straxen.get_resource(map_file, fmt=fmt)
        return straxen.InterpolatingMap(map_data, method=method)

    else:
        raise TypeError("Can't handle map_file except a string or a list")
    
def parse_extension(name):
    """Get the extention from a file name. If zipped or tarred, can contain a dot"""
    split_name = name.split('.')
    if len(split_name) == 2:
        fmt = split_name[-1]
    elif len(split_name) > 2 and 'gz' in name:
        fmt = '.'.join(split_name[-2:])
    else:
        fmt = split_name[-1]
    #log.warning(f'Using {fmt} for unspecified {name}')
    return fmt
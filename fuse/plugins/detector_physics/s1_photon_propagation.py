import numpy as np
import strax
import straxen
import nestpy
import logging

from strax import deterministic_hash
from scipy.interpolate import interp1d

from ...common import loop_uniform_to_pe_arr

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.S1_Signal')
log.setLevel('WARNING')

#Initialize the nestpy random generator
#The seed will be set in the setup function
nest_rng = nestpy.RandomGen.rndm()

@export
class S1PhotonPropagation(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("s1_photons", "microphysics_summary")
    provides = "propagated_s1_photons"
    data_kind = "S1_photons"
    
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

    maximum_recombination_time = straxen.URLConfig(
        type=(int, float),
        help='maximum_recombination_time',
    )

    phase = straxen.URLConfig(
        default="liquid",
        help='phase',
    )

    s1_decay_spread = straxen.URLConfig(
        type=(int, float),
        help='s1_decay_spread',
    )

    s1_decay_time = straxen.URLConfig(
        type=(int, float),
        help='s1_decay_time',
    )

    s1_model_type = straxen.URLConfig(
        help='s1_model_type',
    )

    pmt_circuit_load_resistor = straxen.URLConfig(
        help='pmt_circuit_load_resistor', type=(int, float),
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
    
    s1_pattern_map = straxen.URLConfig(
        cache=True,
        help='s1_pattern_map',
    )
    
    gains = straxen.URLConfig(
        cache=True,
        help='pmt gains',
    )
    
    s1_optical_propagation_spline = straxen.URLConfig(
        cache=True,
        help='s1_optical_propagation_spline',
    )
    
    photon_area_distribution = straxen.URLConfig(
        cache=True,
        help='photon_area_distribution',
    )

    fixed_seed = straxen.URLConfig(
        default=True, type=bool,
        help='Set the random seed from lineage and run_id, or pull the seed from the OS.',
    )
    
    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running S1PhotonPropagation in debug mode")

        if self.fixed_seed:
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            seed = int(hash_string.encode().hex(), 16)
            #Dont know but nestpy seems to have a problem with large seeds
            self.short_seed = int(repr(seed)[-8:])
            nest_rng.set_seed(self.short_seed)

            log.debug(f"Generating random numbers from seed {self.short_seed}")
        else: 
            log.debug(f"Generating random numbers with seed pulled from OS")

        self.turned_off_pmts = np.arange(len(self.gains))[np.array(self.gains) == 0]
        
        if 'nest' in self.s1_model_type: #and (self.nestpy_calc is None):
            log.info('Using NEST for scintillation time without set calculator\n'
                     'Creating new nestpy calculator')
            self.nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())

        self._cached_uniform_to_pe_arr = {}
        self.__uniform_to_pe_arr = self.init_spe_scaling_factor_distributions()

    def compute(self, interactions_in_roi):

        #Just apply this to clusters with photons hitting a PMT
        instruction = interactions_in_roi[interactions_in_roi["n_s1_photon_hits"] > 0]

        if len(instruction) == 0:
            return np.zeros(0, self.dtype)
        
        t = instruction['time']
        x = instruction['x']
        y = instruction['y']
        z = instruction['z']
        n_photons = instruction['photons'].astype(np.int64)
        recoil_type = instruction['nestid']
        positions = np.array([x, y, z]).T  # For map interpolation
        
        # The new way interpolation is written always require a list
        _photon_channels = self.photon_channels(positions=positions,
                                                n_photon_hits=instruction["n_s1_photon_hits"],
                                                )
        
        extra_targs = {}
        if 'nest' in self.s1_model_type:
            extra_targs['n_photons_emitted'] = n_photons
            extra_targs['n_excitons'] = instruction['excitons'].astype(np.int64)
            extra_targs['local_field'] = instruction['e_field']
            extra_targs['e_dep'] = instruction['ed']
            extra_targs['nestpy_calc'] = self.nestpy_calc
            
        _photon_timings = self.photon_timings(t=t,
                                              n_photon_hits=instruction["n_s1_photon_hits"],
                                              recoil_type=recoil_type,
                                              channels=_photon_channels,
                                              positions=positions,
                                              **extra_targs
                                             )
        
        #I should sort by time i guess
        sortind = np.argsort(_photon_timings)
        _photon_channels = _photon_channels[sortind]
        _photon_timings = _photon_timings[sortind]

        #Do i want to save both -> timings with and without pmt transition time spread?
        # Correct for PMT Transition Time Spread (skip for pmt after-pulses)
        # note that PMT datasheet provides FWHM TTS, so sigma = TTS/(2*sqrt(2*log(2)))=TTS/2.35482
        _photon_timings += self.rng.normal(self.pmt_transit_time_mean,
                                            self.pmt_transit_time_spread / 2.35482,
                                            len(_photon_timings)).astype(np.int64)
        
        #Why is this done here and additionally in the get_n_photons function of S1PhotonHits??
        _photon_is_dpe = self.rng.binomial(n=1,
                                            p=self.p_double_pe_emision,
                                            size=len(_photon_timings)).astype(np.bool_)


        _photon_gains = self.gains[_photon_channels] \
            * loop_uniform_to_pe_arr(self.rng.random(len(_photon_channels)), _photon_channels, self.__uniform_to_pe_arr)

        # Add some double photoelectron emission by adding another sampled gain
        n_double_pe = _photon_is_dpe.sum()
        _photon_gains[_photon_is_dpe] += self.gains[_photon_channels[_photon_is_dpe]] \
            * loop_uniform_to_pe_arr(self.rng.random(n_double_pe), _photon_channels[_photon_is_dpe], self.__uniform_to_pe_arr) 

        
        result = np.zeros(_photon_channels.shape[0], dtype = self.dtype)
        result["channel"] = _photon_channels
        result["time"] = _photon_timings
        result["endtime"] = result["time"]
        result["dpe"] = _photon_is_dpe
        result["photon_gain"] = _photon_gains
        
        return result
    
    
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
                                         self.rng.choice(
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
            _photon_timings += self.rng.exponential(self.s1_decay_time, _n_hits_total).astype(np.int64)
            _photon_timings += self.rng.normal(0, self.s1_decay_spread, _n_hits_total).astype(np.int64)

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
                       self.rng.choice(scint_time, counts, replace=False).astype(np.int64)

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
        z_rand = np.array([z_positions, self.rng.random(len(channels))]).T

        is_top = channels < self.n_top_pmts
        prop_time[is_top] = self.s1_optical_propagation_spline(z_rand[is_top], map_name='top')

        is_bottom = channels >= self.n_top_pmts
        prop_time[is_bottom] = self.s1_optical_propagation_spline(z_rand[is_bottom], map_name='bottom')

        return prop_time
    

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
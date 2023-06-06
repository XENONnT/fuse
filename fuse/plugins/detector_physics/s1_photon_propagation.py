import numpy as np
import strax
import straxen
import nestpy
import logging

from .photon_propagation_base import PhotonPropagationBase

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.detector_physics.S1_Signal')
log.setLevel('WARNING')

@export
class S1PhotonPropagation(PhotonPropagationBase):
    
    __version__ = "0.0.0"
    
    depends_on = ("s1_photons", "microphysics_summary")
    provides = "propagated_s1_photons"
    data_kind = "S1_photons"
    
    child_plugin = True

    #Config options specific to S1 simulation
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
    
    s1_pattern_map = straxen.URLConfig(
        cache=True,
        help='s1_pattern_map',
    )
    
    s1_optical_propagation_spline = straxen.URLConfig(
        cache=True,
        help='s1_optical_propagation_spline',
    )
    
    photon_area_distribution = straxen.URLConfig(
        cache=True,
        help='photon_area_distribution',
    )
    
    def setup(self):

        super().setup()

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running S1PhotonPropagation in debug mode")

        if 'nest' in self.s1_model_type: #and (self.nestpy_calc is None):
            log.info('Using NEST for scintillation time without set calculator\n'
                     'Creating new nestpy calculator')
            self.nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())

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
        _photon_timings, _photon_gains, _photon_is_dpe = super().pmt_transition_time_spread(_photon_timings, _photon_channels)
        
        result = super().build_output(_photon_timings, _photon_channels, _photon_gains, _photon_is_dpe)

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
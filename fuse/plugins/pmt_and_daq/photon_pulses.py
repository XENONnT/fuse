import logging
import strax
import straxen
import numpy as np
import numba

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.pmt_and_daq.photon_pulses')

@export
class PulseWindow(strax.Plugin):

    __version__ = "0.0.0"

    depends_on = ("photon_summary")

    provides = ("pulse_windows", "pulse_ids")
    data_kind = {"pulse_windows": "pulse_windows",
                 "pulse_ids" : "propagated_photons"
                }

    dtype_pulse_windows = strax.interval_dtype + [(('pulse_id','identifier for the pulse'), np.int64)]
    dtype_pulse_ids =  [('pulse_id', np.int64),] + strax.time_fields

    dtype = dict()
    dtype["pulse_windows"] = dtype_pulse_windows
    dtype["pulse_ids"] = dtype_pulse_ids

    input_timeout = FUSE_PLUGIN_TIMEOUT

    save_when = strax.SaveWhen.TARGET

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
        help='Show debug informations',
    )

    dt = straxen.URLConfig(
        type=(int),
        help='sample_duration',
    )

    samples_after_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_after_pulse_center',
    )

    samples_to_store_after = straxen.URLConfig(
        type=(int, float),
        help='samples_to_store_after',
    )

    samples_before_pulse_center = straxen.URLConfig(
        type=(int, float),
        help='samples_before_pulse_center',
    )

    samples_to_store_before = straxen.URLConfig(
        type=(int, float),
        help='samples_to_store_before',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running PulseWindow in debug mode")
        else: 
            log.setLevel('WARNING')

        self.pulse_left_extenstion = self.samples_to_store_before + self.samples_before_pulse_center
        #self.pulse_left_extenstion = 0 #Hmm check again what values make sense here
        self.pulse_right_extenstion = self.samples_to_store_after + self.samples_after_pulse_center

    def compute(self, propagated_photons):

        if len(propagated_photons) == 0:
            return {"pulse_windows" : np.zeros(0, self.dtype["pulse_windows"]),
                    "pulse_ids" : np.zeros(0, self.dtype["pulse_ids"])}
        
        single_photon_pulses = np.zeros(len(propagated_photons), dtype=strax.interval_dtype)

        #Can be removed if i sort it in photon summary!!
        propagated_photons = strax.sort_by_time(propagated_photons)
        
        single_photon_pulses["length"] = 22 #get this one from the single photon pmt pulse shape thingy
        single_photon_pulses["dt"] = self.dt
        single_photon_pulses["time"] = propagated_photons["time"]
        single_photon_pulses["channel"] = propagated_photons["channel"]

        photon_pulses, photon_id = concat_overlapping_hits(
            single_photon_pulses,
            (self.pulse_left_extenstion,self.pulse_right_extenstion),
            (0,493), #Set this from the config args
            single_photon_pulses["time"].min(), #This sould also be something different i guess
            single_photon_pulses["time"].max() #This sould also be something different i guess
            )
        photon_pulses = strax.sort_by_time(photon_pulses)

        pulse_ids = np.zeros(len(photon_id), self.dtype["pulse_ids"])
        pulse_ids["pulse_id"] = photon_id
        pulse_ids["time"] = propagated_photons["time"]
        pulse_ids["endtime"] = propagated_photons["endtime"]

        return {"pulse_windows": photon_pulses,
                "pulse_ids" : pulse_ids}


#Modified code taken from strax: 
#https://github.com/AxFoundation/strax/blob/2fb4d1dd7186c81e797aa2773701cf3d693a1d67/strax/processing/hitlets.py#L55C1-L156
def concat_overlapping_hits(hits, extensions, pmt_channels, start, end):
    """
    Function which concatenates hits which may overlap after left and 
    right hit extension. Assumes that hits are sorted correctly.

    Note:
        This function only updates time, and length of the hit.

    :param hits: Hits in records.
    :param extensions: Tuple of the left and right hit extension.
    :param pmt_channels: Tuple of the detectors first and last PMT
    :param start: Startime of the chunk
    :param end: Endtime of the chunk

    :returns:
        array with concataneted hits.
    """
    first_channel, last_channel = pmt_channels
    nchannels = last_channel - first_channel + 1

    # Buffer for concat_overlapping_hits, if specified in 
    # _concat_overlapping_hits numba crashes.
    last_hit_in_channel = np.zeros(nchannels,
                                   dtype=(strax.interval_dtype
                                          + [(('End time of the interval (ns since unix epoch)',
                                               'endtime'), np.int64),
                                             (('pulse_id','identifier for the pulse'), np.int64)
                                               ]))

    pulse_id = 0
    photon_identifiers = np.zeros(len(hits), dtype=np.int64)

    if len(hits):
        hits = _concat_overlapping_hits(
            hits, extensions, first_channel, last_hit_in_channel,photon_identifiers,pulse_id, start, end)
    return hits, photon_identifiers

pulse_dtype = strax.interval_dtype + [(('pulse_id','identifier for the pulse'), np.int64)]

@strax.utils.growing_result(pulse_dtype, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(hits,
                             extensions,
                             first_channel,
                             last_hit_in_channel_buffer,
                             photon_identifiers,
                             pulse_id,
                             chunk_start=0,
                             chunk_end=float('inf'),
                             _result_buffer=None,
                             ):
    buffer = _result_buffer
    res_offset = 0
    

    left_extension, right_extension = extensions
    dt = hits['dt'][0]
    assert np.all(hits['dt'] == dt), 'All hits must have the same dt!'

    for i, hit in enumerate(hits):

        time_with_le = hit['time'] - int(left_extension * hit['dt'])
        endtime_with_re = strax.endtime(hit) + int(right_extension * hit['dt'])
        hit_channel = hit['channel']

        last_hit_in_channel = last_hit_in_channel_buffer[hit_channel - first_channel]
        

        found_no_hit_for_channel_yet = last_hit_in_channel['time'] == 0
        if found_no_hit_for_channel_yet:
            last_hit_in_channel['time'] = max(time_with_le, chunk_start)
            last_hit_in_channel['endtime'] = min(endtime_with_re, chunk_end)
            last_hit_in_channel['channel'] = hit_channel
            last_hit_in_channel['dt'] = dt
            
            last_hit_in_channel["pulse_id"] = pulse_id
            photon_identifiers[i] = last_hit_in_channel["pulse_id"]
            pulse_id += 1

        else:
            hits_overlap_in_channel = last_hit_in_channel['endtime'] >= time_with_le
            if hits_overlap_in_channel:
                last_hit_in_channel['endtime'] = endtime_with_re
                photon_identifiers[i] = last_hit_in_channel['pulse_id']
            else:
                # No, this means we have to save the previous data and update lhc:
                res = buffer[res_offset]
                res['time'] = last_hit_in_channel['time']
                hitlet_length = (last_hit_in_channel['endtime'] - last_hit_in_channel['time'])
                hitlet_length //= last_hit_in_channel['dt']
                res['length'] = hitlet_length
                res['channel'] = last_hit_in_channel['channel']
                res['dt'] = last_hit_in_channel['dt']
                res["pulse_id"] = last_hit_in_channel['pulse_id']
                
                # Updating current last hit:
                last_hit_in_channel['time'] = time_with_le
                last_hit_in_channel['endtime'] = endtime_with_re
                last_hit_in_channel['channel'] = hit_channel
                
                
                last_hit_in_channel["pulse_id"] = pulse_id
                photon_identifiers[i] = last_hit_in_channel["pulse_id"]
                pulse_id += 1
                
                res_offset += 1
                if res_offset == len(buffer):
                    yield res_offset
                    res_offset = 0


    # We went through so now we have to save all remaining hits:
    mask = last_hit_in_channel_buffer['time'] != 0
    for last_hit_in_channel in last_hit_in_channel_buffer[mask]:
        res = buffer[res_offset]
        res['time'] = last_hit_in_channel['time']
        res['channel'] = last_hit_in_channel['channel']
        hitlet_length = (last_hit_in_channel['endtime'] - last_hit_in_channel['time'])
        hitlet_length //= last_hit_in_channel['dt']
        res['length'] = hitlet_length
        res['dt'] = last_hit_in_channel['dt']
        res["pulse_id"] = last_hit_in_channel["pulse_id"]

        res_offset += 1
        if res_offset == len(buffer):
            yield res_offset
            res_offset = 0
    yield res_offset
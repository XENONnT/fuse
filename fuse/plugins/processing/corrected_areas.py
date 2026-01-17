import strax
import straxen

export, __all__ = strax.exporter()


@export
class CorrectedAreasMC(straxen.CorrectedAreas):
    """Corrected areas plugin for MC data.

    This plugin overwrites the (alt_)cs1 and (alt_)cs2 fields with the
    not-time- corrected values as the effects of time dependent single-
    electron gain, extraction efficiency and photo ionization are not
    simulated in fuse.

    Applying corrections for these effects to simulated data would lead
    to incorrect results. This plugins should only be used for simulated
    data!
    """

    __version__ = "0.0.2"
    child_plugin = True

    def compute(self, events):
        result = super().compute(events)

        result["cs1"] = result["cs1_wo_timecorr"]
        result["alt_cs1"] = result["alt_cs1_wo_timecorr"]

        if "cs2_wo_timecorr" in result.dtype.names:
            result["cs2"] = result["cs2_wo_timecorr"]
            result["alt_cs2"] = result["alt_cs2_wo_timecorr"]
        elif "cs2_w_bias_xy" in result.dtype.names:
            result["cs2"] = result["cs2_w_bias_xy"]
            result["alt_cs2"] = result["alt_cs2_w_bias_xy"]
        else:
            raise RuntimeError("Please check your straxen compatibility")

        return result

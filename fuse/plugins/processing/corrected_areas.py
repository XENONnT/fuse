import strax
import straxen

export, __all__ = strax.exporter()


@export
class CorrectedAreasMC(straxen.CorrectedAreas):
    """Corrected areas plugin for MC data.

    This plugin overwrites the cs1 and cs2 fields with the not-time-
    corrected values as the effects are not simulated in the MC.
    """

    __version__ = "0.0.1"
    child_plugin = True

    def compute(self, events):
        result = super().compute(events)
        result["cs1"] = result["cs1_wo_timecorr"]
        result["cs2"] = result["cs2_wo_timecorr"]

        return result

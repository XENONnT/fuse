import strax

from ...vertical_merger_plugin import VerticalMergerPlugin
from ...dtypes import propagated_photons_fields
from immutabledict import immutabledict


export, __all__ = strax.exporter()


@export
class PhotonSummary(VerticalMergerPlugin):
    """Plugin that concatenates propagated photons for S1, S2 and PMT
    afterpulses."""

    depends_on = ("propagated_s2_photons", "propagated_s1_photons", "pmt_afterpulses")
    __version__ = "0.1.0"

    provides = ("photon_summary", "pi_absorbed_summary")
    data_kind = {
        "photon_summary": "propagated_photons", 
        "pi_absorbed_summary": "non_propagated_photons"
    }

    save_when = immutabledict(
        photon_summary=strax.SaveWhen.TARGET,
        pi_absorbed_summary=strax.SaveWhen.TARGET
    )

    def infer_dtype(self):
        dtype = dict()
        dtype["photon_summary"] = propagated_photons_fields + strax.time_fields
        dtype["pi_absorbed_summary"] = propagated_photons_fields + strax.time_fields
        return dtype

    def compute(self, **kwargs):
        result = super().compute(**kwargs)

        result_photon_summary = result[~result['pi_absorbed']]
        result_pi_absorbed_summary = result[result['pi_absorbed']]

        strax.sort_by_time(result_photon_summary)
        strax.sort_by_time(result_pi_absorbed_summary)

        return {
            "photon_summary": result_photon_summary,
            "pi_absorbed_summary": result_pi_absorbed_summary
        }

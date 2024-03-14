import strax

from ...vertical_merger_plugin import VerticalMergerPlugin

export, __all__ = strax.exporter()


@export
class PhotonSummary(VerticalMergerPlugin):
    """Plugin that concatenates propagated photons for S1, S2 and PMT
    afterpulses."""

    depends_on = ("propagated_s2_photons", "propagated_s1_photons", "pmt_afterpulses")

    provides = "photon_summary"
    data_kind = "propagated_photons"
    __version__ = "0.1.0"

import strax

from ....vertical_merger_plugin import VerticalMergerPlugin

export, __all__ = strax.exporter()


@export
class DriftedElectronsMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    drift plugins."""

    depends_on = ("drifted_electrons", "drifted_delayed_electrons")

    provides = "merged_drifted_electrons"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.1"


@export
class ExtractedElectronsMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    extraction plugins."""

    depends_on = ("extracted_electrons", "extracted_delayed_electrons")

    provides = "merged_extracted_electrons"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.1"


@export
class SecondaryScintillationPhotonsMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    secondary scintillation plugins."""

    depends_on = ("s2_photons", "delayed_electrons_s2_photons")

    provides = "merged_s2_photons"
    data_kind = "individual_electrons"
    __version__ = "0.0.1"


@export
class SecondaryScintillationPhotonSumMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    secondary scintillation plugins."""

    depends_on = ("s2_photons_sum", "delayed_electrons_s2_photons_sum")

    provides = "merged_s2_photons_sum"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.1"


@export
class ElectronTimingMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    timing plugins."""

    depends_on = ("electron_time", "delayed_electrons_time")

    provides = "merged_electron_time"
    data_kind = "individual_electrons"
    __version__ = "0.0.1"


@export
class MicrophysicsSummaryMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed electron
    secondary scintillation plugins."""

    depends_on = ("microphysics_summary", "photo_ionization_electrons")

    provides = "merged_microphysics_summary"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.1"


@export
class S1PhotonHitsMerger(VerticalMergerPlugin):
    """Plugin which concatenates the output of the regular and delayed s1
    photon hits plugins."""

    depends_on = ("s1_photon_hits", "delayed_s1_photon_hits")

    provides = "merged_s1_photon_hits"
    data_kind = "interactions_in_roi"
    __version__ = "0.0.2"

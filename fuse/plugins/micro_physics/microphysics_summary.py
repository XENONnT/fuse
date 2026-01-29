import strax

export, __all__ = strax.exporter()


@export
class MicroPhysicsSummary(strax.MergeOnlyPlugin):
    """MergeOnlyPlugin that summarizes the fuse microphysics simulation results
    into a single output."""

    depends_on = (
        "microphysics_summary",
        "quanta",
        "electric_field_values",
    )
    rechunk_on_save = False
    save_when = strax.SaveWhen.ALWAYS
    provides = "microphysics_summary_AHA"
    __version__ = "0.1.0"

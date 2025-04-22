import strax

export, __all__ = strax.exporter()


@export
class FastsimEvents(strax.MergeOnlyPlugin):
    """MergeOnlyPlugin that summarizes fastsim events and corrections into a
    single output."""

    depends_on = (
        "fastsim_events_uncorrected",
        "fastsim_corrections",
    )
    rechunk_on_save = False
    save_when = strax.SaveWhen.TARGET
    provides = "fastsim_events"
    __version__ = "0.1.0"

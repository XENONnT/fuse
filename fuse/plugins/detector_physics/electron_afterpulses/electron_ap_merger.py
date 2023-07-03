import strax

from ....vertical_merger_plugin import VerticalMergerPlugin

export, __all__ = strax.exporter()

@export
class ElectronAfterPulsesSummary(VerticalMergerPlugin):
    """
    Plugin which concatenates the output of the photo-electric and photo-ionization afterpulse simulation
    """
    
    depends_on = ()
    
    provides = 'electron_ap_summary'
    data_kind = "interactions_in_roi"
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False
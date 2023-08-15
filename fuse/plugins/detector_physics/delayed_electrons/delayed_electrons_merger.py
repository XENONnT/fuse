import strax

from ....vertical_merger_plugin import VerticalMergerPlugin

export, __all__ = strax.exporter()

@export
class DriftedElectronsMerger(VerticalMergerPlugin):
    """
    Plugin which concatenates the output of the regular and delayed electron drift plugins
    """
    
    depends_on = ("drifted_electrons", "drifted_delayed_electrons")
    
    provides = 'merged_drifted_electrons'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class ExtractedElectronsMerger(VerticalMergerPlugin):
    """
    Plugin which concatenates the output of the regular and delayed electron extraction plugins
    """
    
    depends_on = ("extracted_electrons", "extracted_delayed_electrons")
    
    provides = 'merged_extracted_electrons'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class SecondaryScintillationPhotonsMerger(VerticalMergerPlugin):
    """
    Plugin which concatenates the output of the regular and delayed electron secondary scintillation plugins
    """
    
    depends_on = ("s2_photons", "delayed_electrons_s2_photons")
    
    provides = 'merged_s2_photons'
    data_kind = 'individual_electrons'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class SecondaryScintillationPhotonSumMerger(VerticalMergerPlugin):
    """
    Plugin which concatenates the output of the regular and delayed electron secondary scintillation plugins
    """
    
    depends_on = ("s2_photons_sum", "delayed_electrons_s2_photons_sum")
    
    provides = 'merged_s2_photons_sum'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False



#The following plugins are used for simulations without delayed electrons
#Using separate plugins enables us to use the same S2PhotonPropagation plugin for both
@export
class DriftedElectronsRename(VerticalMergerPlugin):
    """
    Plugin which basically renames the output of the regular electron drift plugin
    """
    
    depends_on = ("drifted_electrons")
    
    provides = 'merged_drifted_electrons'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class ExtractedElectronsRename(VerticalMergerPlugin):
    """
    Plugin which basically renames the output of the regular electron extraction plugin
    """
    
    depends_on = ("extracted_electrons")
    
    provides = 'merged_extracted_electrons'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class SecondaryScintillationPhotonsRename(VerticalMergerPlugin):
    """
    Plugin which basically renames the output of the regular secondary scintillation plugin
    """
    
    depends_on = ("s2_photons")
    
    provides = 'merged_s2_photons'
    data_kind = 'individual_electrons'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

@export
class SecondaryScintillationPhotonSumRename(VerticalMergerPlugin):
    """
    Plugin which basically renames the output of the regular secondary scintillation plugin
    """
    
    depends_on = ("s2_photons_sum")
    
    provides = 'merged_s2_photons_sum'
    data_kind = 'interactions_in_roi'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

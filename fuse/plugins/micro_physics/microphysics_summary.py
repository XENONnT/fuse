import strax

export, __all__ = strax.exporter()

from ...common import FUSE_PLUGIN_TIMEOUT

@export
class MicroPhysicsSummary(strax.MergeOnlyPlugin):
    """
    Plugin which summarizes the MicroPhysics simulation into a single output
    """

    depends_on = ['interactions_in_roi',
                  'quanta',
                  'electric_field_values',
                  ]
    save_when = strax.SaveWhen.ALWAYS
    provides = 'microphysics_summary'
    __version__ = '0.0.0'

    #Forbid rechunking
    rechunk_on_save = False

    input_timeout = FUSE_PLUGIN_TIMEOUT
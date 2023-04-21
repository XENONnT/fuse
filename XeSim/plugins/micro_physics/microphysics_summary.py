import strax

class MicroPhysicsSummary(strax.MergeOnlyPlugin):
    """
    Plugin which summarizes the MicroPhysics simulation into a single output
    """

    depends_on = ['clustered_interactions',
                  'quanta',
                  'electric_field_values',
                  ]
    save_when = strax.SaveWhen.ALWAYS
    provides = 'microphysics_summary'
    __version__ = '0.0.0'

    def compute(self, **kwargs):
        
        microphysics_summary = super().compute(**kwargs)

        return microphysics_summary
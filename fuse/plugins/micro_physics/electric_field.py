import strax
import numpy as np
import logging
import straxen

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.electric_field')
log.setLevel('WARNING')

@export
class ElectricField(strax.Plugin):
    """
    Plugin that calculates the electric field values for the detector.
    """

    __version__ = "0.0.0"

    depends_on = ("clustered_interactions",)
    provides = "electric_field_values"
    data_kind = "clustered_interactions"

    #Forbid rechunking
    rechunk_on_save = False

    dtype = [
        ('e_field', np.int64),
        *strax.time_fields
    ]

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,
        help='Show debug informations',
    )

    efield_map = straxen.URLConfig(
        cache=True,
        help='electric field map',
    )

    def setup(self):

        if self.debug:
            log.setLevel('DEBUG')
            log.debug("Running ElectricField in debug mode")

    def compute(self, clustered_interactions):
        """
        Calculate the electric field values for the given clustered interactions.

        Args:
            clustered_interactions (numpy.ndarray): array of clustered interactions.

        Returns:
            numpy.ndarray: array of electric field values.
        """
        if len(clustered_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        electric_field_array = np.zeros(len(clustered_interactions), dtype=self.dtype)
        electric_field_array['time'] = clustered_interactions['time']
        electric_field_array['endtime'] = clustered_interactions['endtime']

        r = np.sqrt(clustered_interactions['x'] ** 2 + clustered_interactions['y'] ** 2)
        positions = np.stack((r, clustered_interactions['z']), axis=1)
        electric_field_array['e_field'] = self.efield_map(positions)

        return electric_field_array


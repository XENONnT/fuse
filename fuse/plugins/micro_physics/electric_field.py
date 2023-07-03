import strax
import numpy as np
import logging
import straxen

from ...common import FUSE_PLUGIN_TIMEOUT

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

    depends_on = ("interactions_in_roi",)
    provides = "electric_field_values"
    data_kind = "interactions_in_roi"

    #Forbid rechunking
    rechunk_on_save = False

    save_when = strax.SaveWhen.TARGET

    input_timeout = FUSE_PLUGIN_TIMEOUT

    dtype = [
        ('e_field', np.int64),
        *strax.time_fields
    ]

    #Config options
    debug = straxen.URLConfig(
        default=False, type=bool,track=False,
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

    def compute(self, interactions_in_roi):
        """
        Calculate the electric field values for the given clustered interactions.

        Args:
            interactions_in_roi (numpy.ndarray): array of clustered interactions.

        Returns:
            numpy.ndarray: array of electric field values.
        """
        if len(interactions_in_roi) == 0:
            return np.zeros(0, dtype=self.dtype)

        electric_field_array = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        electric_field_array['time'] = interactions_in_roi['time']
        electric_field_array['endtime'] = interactions_in_roi['endtime']

        r = np.sqrt(interactions_in_roi['x'] ** 2 + interactions_in_roi['y'] ** 2)
        positions = np.stack((r, interactions_in_roi['z']), axis=1)
        electric_field_array['e_field'] = self.efield_map(positions)

        # Clip negative values to 0
        n_negative_values = np.sum(electric_field_array['e_field'] < 0)
        if n_negative_values > 0:
            log.warning(f"Found {n_negative_values} negative electric field values. Clipping to 0.")
        electric_field_array['e_field'] = np.clip(electric_field_array['e_field'], 0, None)

        return electric_field_array


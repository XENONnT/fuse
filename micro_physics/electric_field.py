import strax
import epix
import numpy as np


@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug information"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder")
)
class ElectricField(strax.Plugin):
    """
    Plugin that calculates the electric field values for the detector.
    """

    __version__ = "0.0.0"

    depends_on = ("clustered_interactions",)
    provides = "electric_field_values"
    data_kind = "clustered_interactions"

    dtype = [
        ('e_field', np.int64),
        *strax.time_fields
    ]

    def setup(self):
        """
        Do the volume cuts here.

        Initialize the detector config.
        """
        self.detector_config = epix.init_detector(
            self.config['Detector'].lower(),
            self.config['DetectorConfigOverride']
        )

    def compute(self, clustered_interactions):
        """
        Calculate the electric field values for the given clustered interactions.

        Args:
            clustered_interactions (numpy.ndarray): array of clustered interactions.

        Returns:
            numpy.ndarray: array of electric field values.
        """
        electric_field_array = np.zeros(len(clustered_interactions), dtype=self.dtype)
        electric_field_array['time'] = clustered_interactions['time']
        electric_field_array['endtime'] = clustered_interactions['endtime']

        efields = electric_field_array['e_field']

        for volume in self.detector_config:
            if isinstance(volume.electric_field, (float, int)):
                volume_id_mask = clustered_interactions['vol_id'] == volume.volume_id
                efields[volume_id_mask] = volume.electric_field
            else:
                efields = volume.electric_field(
                    clustered_interactions['x'],
                    clustered_interactions['y'],
                    clustered_interactions['z']
                )

        electric_field_array['e_field'] = efields

        return electric_field_array

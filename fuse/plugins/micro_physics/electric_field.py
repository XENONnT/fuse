import strax
import epix
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

    detector = straxen.URLConfig(
        default="XENONnT", 
        help='Detector to be used. Has to be defined in epix.detectors',
    )

    detector_config_override = straxen.URLConfig(
        default=None, 
        help='Config file to overwrite default epix.detectors settings; see examples in the configs folder',
    )

    def setup(self):
        """
        Do the volume cuts here.

        Initialize the detector config.
        """
        self.detector_config = epix.init_detector(
            self.detector.lower(),
            self.detector_config_override
        )

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


import strax
import numpy as np
import logging
import straxen

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('fuse.micro_physics.electric_field')

@export
class ElectricField(FuseBasePlugin):
    """
    Plugin that calculates the electric field values for the cluster position.
    """

    __version__ = "0.2.2"

    depends_on = ("interactions_in_roi",)
    provides = "electric_field_values"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (("Electric field value at the cluster position [V/cm]", "e_field"), np.float32),
        *strax.time_fields
    ]

    #Config options
    #Field map not yet in simulation config file!
    efield_map = straxen.URLConfig(
        default = 'itp_map://resource://'
                  'fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz?'
                  '&fmt=json.gz'
                  '&method=RegularGridInterpolator',
        cache=True,
        help='Map of the electric field in the detector',
    )

    def compute(self, interactions_in_roi):
        
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

        # Clip NaN values to 0
        n_nan_values = np.sum(np.isnan(electric_field_array['e_field']))
        if n_nan_values > 0:
            log.warning(f"Found {n_nan_values} NaN electric field values. Clipping to 0.")
        electric_field_array['e_field'] = np.nan_to_num(electric_field_array['e_field'])

        return electric_field_array


import strax
import straxen
import numpy as np
from ...common import FUSE_PLUGIN_TIMEOUT

export, __all__ = strax.exporter()


@export
class Truth(strax.Plugin):
    __version__ = '0.1.0'

    provides = "truth"
    depends_on = ("quanta", "electric_field_values", "interactions_in_roi")
    data_kind = "interactions_in_roi"

    rechunk_on_save = False
    save_when = strax.SaveWhen.TARGET
    input_timeout = FUSE_PLUGIN_TIMEOUT

    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float32),
             ('nestid', np.int8),
             ('A', np.int8),
             ('Z', np.int8),
             ('evtid', np.int32),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int8),
             ('create_S2', np.bool8),
             ('photons', np.int32),
             ('electrons', np.int32),
             ('excitons', np.int32),
             ]

    dtype = dtype + strax.time_fields

    # Config options
    debug = straxen.URLConfig(
        default=False, type=bool, track=False,
        help='Show debug informations',
    )

    def compute(self, interactions_in_roi):
        if len(interactions_in_roi) > 0:
            result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
            for dt in self.dtype.names:
                result[dt] = interactions_in_roi[dt]
        else:
            result = np.zeros(0, dtype=self.dtype)

        return result

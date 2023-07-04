import unittest
import fuse
import strax
import straxen
from straxen import URLConfig
import numpy as np

@URLConfig.register('DummyMap')
def return_dummy_map(data, value):
    
    return fuse.common.DummyMap(const = value)

input_dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int64),
             ('create_S2', np.bool8),
            ]
    
input_dtype = input_dtype + strax.time_fields

class TestElectricField(unittest.TestCase):

    def setUp(self):

        self.test_context = fuse.context.microphysics_context("/scratch/midway2/hschulze/fuse_data/")

        self.test_context.set_config({"debug": True,
                                      "efield_map": "DummyMap://format://?&value=1"
                                      })

        self.plugin = self.test_context.get_single_plugin("00000", "electric_field_values")

    def test_plugin(self):

        plugin_intput = np.zeros(20, dtype=input_dtype)

        plugin_output = self.plugin.compute(plugin_intput)

        self.assertTrue(np.all(plugin_output["e_field"] == 1))

if __name__ == '__main__':
    unittest.main()
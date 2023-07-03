import strax
from strax import Plugin, SaveWhen

import numpy as np
from itertools import groupby

class VerticalMergerPlugin(Plugin):
    "Plugin that concatenates data from the dependencies along the fist axis"

    save_when = SaveWhen.NEVER
    
    def infer_dtype(self):
        incoming_dtypes = [self.deps[d].dtype_for(d) for d in sorted(self.depends_on)]
        
        eq = self.all_equal(incoming_dtypes)
        if eq == False:
            raise ValueError("VerticalMergerPlugin can only merge data "
                             "with the same dtype! "
                            )

        return incoming_dtypes[0]
        
    
    def compute(self, **kwargs):

        merged_data = np.concatenate([kwargs[x] for x in kwargs])

        #Sort everything by time
        sortind = np.argsort(merged_data["time"])
        merged_data = merged_data[sortind]

        return merged_data
    
    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

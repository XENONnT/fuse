import strax

import numpy as np
from itertools import groupby

from .plugin import FuseBasePlugin


class VerticalMergerPlugin(FuseBasePlugin):
    "Plugin that concatenates data from the dependencies along the fist axis"

    save_when = strax.SaveWhen.TARGET

    def setup(self):
        super().setup()

    def infer_dtype(self):
        incoming_dtypes = [self.deps[d].dtype_for(d) for d in sorted(self.depends_on)]

        eq = self.all_equal(incoming_dtypes)
        if not eq:
            raise ValueError("VerticalMergerPlugin can only merge data with the same dtype!")

        return incoming_dtypes[0]

    def compute(self, **kwargs):
        merged_data = np.concatenate([kwargs[x] for x in kwargs])

        return strax.sort_by_time(merged_data)

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

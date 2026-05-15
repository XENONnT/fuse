__version__ = "1.6.3"

# Install the fast InterpolatingMap subclass BEFORE the plugins import
# (which triggers URLConfig protocol registration). After this point any
# `itp_map://` or `pattern_map://` URL resolved by straxen builds a
# FastInterpolatingMap instance, not the vanilla one.
from . import fast_interpolation
fast_interpolation.install_global_patch()

from . import dtypes
from .dtypes import *

from . import plugins
from .plugins import *

from .vertical_merger_plugin import *

from .context import *
from .context_utils import *

"""Faster drop-in replacement for `straxen.InterpolatingMap`.

Same public API as the upstream class. Internally, every
``WeightedNearestNeighbors`` sub-interpolator (the default and the one
used by every map fuse loads) is replaced by a variant that keeps
scipy's ``cKDTree.query`` but fuses the post-query arithmetic
(gather + weights + reduction) into a single numba kernel with no
intermediate allocation.

Two reduction paths from the original ``InterpolateAndExtrapolate.__call__``
are reproduced exactly:

  * **3D values, array-valued** -> matches
    ``np.einsum("ijk, ij->ik", values, weights / weights.sum(...))``
  * **scalar values** -> matches
    ``np.average(values, weights=weights, axis=-1)``

Float-op order is preserved within each output row (ascending j on the
contracted axis), so output agrees with vanilla to within a single ULP
of round-off.
"""
import numpy as np
import numba
import straxen


@numba.njit(cache=True, nogil=True)
def _wnn_arrayvalued(values, indices, distances, out):
    # values    : (N_grid, n_pmt) float64  - assumed C-contiguous
    # indices   : (N_q, k)        int64
    # distances : (N_q, k)        float64
    # out       : (N_q, n_pmt)    float64 - written
    #
    # Loop order (i outer, j middle, m inner): each (i, j) step reads one
    # contiguous row `values[indices[i, j], :]` and accumulates into one
    # contiguous output row `out[i, :]`. Both arrays stay in cache for
    # the n_pmt ~ 500 case. Weight computation hoisted out of the m loop.
    n_q, k = indices.shape
    n_pmt = values.shape[1]
    for i in range(n_q):
        wsum = 0.0
        for j in range(k):
            d = distances[i, j]
            if d < 1e-6:
                d = 1e-6
            wsum += 1.0 / d
        inv_wsum = 1.0 / wsum
        for m in range(n_pmt):
            out[i, m] = 0.0
        for j in range(k):
            d = distances[i, j]
            if d < 1e-6:
                d = 1e-6
            w_norm = (1.0 / d) * inv_wsum
            row = indices[i, j]
            for m in range(n_pmt):
                out[i, m] += values[row, m] * w_norm


@numba.njit(cache=True, nogil=True)
def _wnn_scalar(values, indices, distances, out):
    # values    : (N_grid,)  float64
    # indices   : (N_q, k)   int64
    # distances : (N_q, k)   float64
    # out       : (N_q,)     float64
    # Matches np.average(values, weights=1/clip(distances, 1e-6, inf), axis=-1):
    #   num = sum_j (values[idx[i,j]] * w[i,j])   (ascending j)
    #   den = sum_j w[i,j]                        (ascending j)
    #   out[i] = num / den
    n_q, k = indices.shape
    for i in range(n_q):
        num = 0.0
        den = 0.0
        for j in range(k):
            d = distances[i, j]
            if d < 1e-6:
                d = 1e-6
            w = 1.0 / d
            num += values[indices[i, j]] * w
            den += w
        out[i] = num / den


class FastInterpolateAndExtrapolate(straxen.InterpolateAndExtrapolate):
    """Drop-in replacement for ``straxen.InterpolateAndExtrapolate``.

    Re-uses the parent's ``cKDTree`` and ``values`` and only overrides
    ``__call__`` to route through the fused numba kernels above.
    """

    @classmethod
    def from_vanilla(cls, base):
        new = cls.__new__(cls)
        new.kdtree = base.kdtree
        new.values = np.ascontiguousarray(base.values)
        new.neighbours_to_use = base.neighbours_to_use
        new.array_valued = base.array_valued
        new.rotated = getattr(base, "rotated", False)
        if base.array_valued:
            new.n_dim = base.n_dim
        return new

    def __call__(self, points):
        points = np.asarray(points)
        # Match vanilla: rotate query points up front for maps built in
        # the perp-wires frame.
        if self.rotated:
            assert points.shape[1] == 2, "rotated maps only support 2D input"
            points = np.array(straxen.rotate_perp_wires(points[:, 0], points[:, 1])).T

        if self.array_valued:
            result = np.empty((len(points), self.n_dim))
        else:
            result = np.empty(len(points))
        valid = np.all(np.isfinite(points), axis=-1)
        result[~valid] = float("nan")

        pts_valid = points[valid]
        if pts_valid.shape[0] == 0:
            return result

        distances, indices = self.kdtree.query(pts_valid, self.neighbours_to_use)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        distances = np.ascontiguousarray(distances, dtype=np.float64)

        if self.array_valued and self.values.ndim == 2:
            out_buf = np.empty((pts_valid.shape[0], self.values.shape[1]), dtype=np.float64)
            _wnn_arrayvalued(self.values, indices, distances, out_buf)
        elif (not self.array_valued) and self.values.ndim == 1:
            out_buf = np.empty(pts_valid.shape[0], dtype=np.float64)
            _wnn_scalar(self.values, indices, distances, out_buf)
        else:
            # Unsupported shape (rare). Fall back to the parent's path.
            return super().__call__(points)

        result[valid] = out_buf
        return result


class FastInterpolatingMap(straxen.InterpolatingMap):
    """Drop-in replacement for ``straxen.InterpolatingMap``.

    After the parent ``__init__`` builds the sub-interpolators normally,
    each ``InterpolateAndExtrapolate`` (the ``WeightedNearestNeighbors``
    path) is promoted to its fast twin in place. Other methods
    (``RegularGridInterpolator``, ``RectBivariateSpline``,
    ``Linear1Din2D``) are left untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, sub in list(self.interpolators.items()):
            if type(sub) is straxen.InterpolateAndExtrapolate:
                self.interpolators[name] = FastInterpolateAndExtrapolate.from_vanilla(sub)


def install_global_patch():
    """Install ``FastInterpolatingMap`` as the global ``straxen.InterpolatingMap``.

    Both ``straxen.InterpolatingMap`` and ``straxen.itp_map.InterpolatingMap``
    are replaced so the ``itp_map://`` and ``pattern_map://`` URLConfig
    protocols (which look up ``straxen.InterpolatingMap`` lazily at
    URL-resolution time) build instances of the fast subclass instead.
    """
    straxen.InterpolatingMap = FastInterpolatingMap
    try:
        import straxen.itp_map as _itp
        _itp.InterpolatingMap = FastInterpolatingMap
    except Exception:
        pass

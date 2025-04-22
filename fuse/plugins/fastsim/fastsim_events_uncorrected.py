import strax
import numpy as np
import straxen
import logging
from scipy.interpolate import interp1d

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.fastsim.fastsim_s1")


@export
class FastsimEventsUncorrected(FuseBasePlugin):
    """
    Plugin to simulate S1 and (alt) S2 areas from photon hits and electrons extracted

    """
    __version__ = "0.0.1"

    depends_on = ("fastsim_macro_clusters",)
    provides = "fastsim_events_uncorrected"
    data_kind = "fastsim_events"
    dtype = [
                (("S1 area, uncorrected [PE]", "s1_area"), np.float32),
                (("S2 area, uncorrected [PE]", "s2_area"), np.float32),
                (("Alternate S2 area, uncorrected [PE]", "alt_s2_area"), np.float32),
                (("Sum of S2 areas in event, uncorrected [PE]", "s2_sum"), np.float32),
                (("Number of S2s in event", "multiplicity"), np.int32),
                (("Drift time between main S1 and S2 [ns]", "drift_time"), np.float32),
                (("Drift time using alternate S2 [ns]", "alt_s2_interaction_drift_time"), np.float32),
                (("Main S2 reconstructed X position, uncorrected [cm]", "s2_x"), np.float32),
                (("Main S2 reconstructed Y position, uncorrected [cm]", "s2_y"), np.float32),
                (("Alternate S2 reconstructed X position, uncorrected [cm]", "alt_s2_x"), np.float32),
                (("Alternate S2 reconstructed Y position, uncorrected [cm]", "alt_s2_y"), np.float32),
                (("Main interaction r-position with observed position [cm]", "r_naive"), np.float32),
                (("Alternate interaction r-position with observed position [cm]", "alt_s2_r_naive"), np.float32),
                (("Main interaction z-position with observed position [cm]", "z_naive"), np.float32),
                (("Alternate interaction z-position with observed position [cm]", "alt_s2_z_naive"), np.float32),
            ] + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    photon_area_distribution = straxen.URLConfig(
        default="simple_load://resource://simulation_config://"
                "SIMULATION_CONFIG_FILE.json?"
                "&key=photon_area_distribution"
                "&fmt=csv",
        cache=True,
        help="Photon area distribution",
    )

    s2_secondary_sc_gain_mc = straxen.URLConfig(
        default="take://resource://"
                "SIMULATION_CONFIG_FILE.json?&fmt=json"
                "&take=s2_secondary_sc_gain",
        type=(int, float),
        cache=True,
        help="Secondary scintillation gain [PE/e-]",
    )

    se_gain_from_map = straxen.URLConfig(
        default="take://resource://"
                "SIMULATION_CONFIG_FILE.json?&fmt=json"
                "&take=se_gain_from_map",
        cache=True,
        help="Boolean indication if the secondary scintillation gain is taken from a map",
    )

    se_gain_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
                "SIMULATION_CONFIG_FILE.json?"
                "&key=se_gain_map"
                "&fmt=json",
        cache=True,
        help="Map of the single electron gain",
    )

    s2_correction_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
                "SIMULATION_CONFIG_FILE.json?"
                "&key=s2_correction_map"
                "&fmt=json",
        cache=True,
        help="S2 correction map",
    )

    p_double_pe_emission = straxen.URLConfig(
        default="take://resource://"
                "SIMULATION_CONFIG_FILE.json?&fmt=json"
                "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )
    @staticmethod
    def average_spe_distribution(spe_shapes):
        """
            We take the spe distribution from all channels and take the average to be our spe distribution to draw
            photon areas from
        """
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'].values  # / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            grid_cdf = np.linspace(0, 1, 2001)
            # For Inverse_transform_sampling methode
            grid_scale = interp1d(cdf, scaled_bins,
                                  bounds_error=False,
                                  fill_value=(scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)
        spe_distribution = np.mean(uniform_to_pe_arr, axis=0)

        return spe_distribution

    @staticmethod
    def get_s1_area_with_spe(spe_dist, num_photons):
        """
            :params: spe_distribution, the spe distribution to draw photon areas from
            :params: num_photons, number of photons to draw from spe distribution
        """
        return np.sum(spe_dist[(np.random.random(num_photons) * len(spe_dist)).astype(np.int64)])

    def compute(self, fastsim_macro_clusters):
        eventids = np.unique(fastsim_macro_clusters['eventid'])
        result = np.empty(len(eventids), dtype=self.dtype)
        for name in result.dtype.names:
            if np.issubdtype(result.dtype[name], np.floating):
                result[name].fill(np.nan)
        mean_photon_area_distribution = self.average_spe_distribution(self.photon_area_distribution)
        for i, eventid in enumerate(eventids):
            these_clusters = fastsim_macro_clusters[fastsim_macro_clusters['eventid'] == eventid]

            result[i]["time"] = these_clusters[0]["time"]
            result[i]["endtime"] = these_clusters[0]["endtime"]

            photons = np.sum(these_clusters['n_s1_photon_hits']) * (1 + self.p_double_pe_emission)
            result["s1_area"][i] = self.get_s1_area_with_spe(mean_photon_area_distribution, photons.astype(np.int64))

            cluster_info = []
            for cluster in these_clusters:
                s2_area = cluster['sum_s2_photons'] * (1 + self.p_double_pe_emission)
                cluster_info.append((s2_area, cluster))

            # Sort the clusters by s2_area in descending order
            cluster_info_sorted = sorted(cluster_info, key=lambda x: x[0], reverse=True)

            # Assign the highest and second-highest s2_area and drift time values
            if len(cluster_info_sorted) > 0:
                s2_areas = [info[0] for info in cluster_info_sorted]
                result[i]['s2_sum'] = np.sum(s2_areas)
                result[i]['s2_area'] = cluster_info_sorted[0][0]
                result[i]['drift_time'] = cluster_info_sorted[0][1]['drift_time_mean']
                result[i]['s2_x'] = cluster_info_sorted[0][1]['x_obs']
                result[i]['s2_y'] = cluster_info_sorted[0][1]['y_obs']
                result[i]['z_naive'] = cluster_info_sorted[0][1]['z_obs']

            if len(cluster_info_sorted) > 1:
                result[i]['alt_s2_area'] = cluster_info_sorted[1][0]
                result[i]['alt_s2_interaction_drift_time'] = cluster_info_sorted[1][1]['drift_time_mean']
                result[i]['alt_s2_x'] = cluster_info_sorted[1][1]['x_obs']
                result[i]['alt_s2_y'] = cluster_info_sorted[1][1]['y_obs']
                result[i]['alt_s2_z_naive'] = cluster_info_sorted[1][1]['z_obs']

            result[i]["multiplicity"] = len(cluster_info_sorted)

        result['r_naive'] = np.sqrt(result['s2_x'] ** 2 + result['s2_y'] ** 2)
        result['alt_s2_r_naive'] = np.sqrt(result['alt_s2_x'] ** 2 + result['alt_s2_y'] ** 2)

        return result

    def get_s2_light_yield(self, positions):
        """Calculate s2 light yield...

        Args:
            positions: 2d array of positions (floats) returns array of floats (mean expectation)
        """

        if self.se_gain_from_map:
            sc_gain = self.se_gain_map(positions)
        else:
            # calculate it from MC pattern map directly if no "se_gain_map" is given
            sc_gain = self.s2_correction_map(positions)
            sc_gain *= self.s2_secondary_sc_gain_mc

        # Depending on if you use the data driven or mc pattern map for light yield for S2
        # The shape of n_photon_hits will change. Mc needs a squeeze
        if len(sc_gain.shape) != 1:
            sc_gain = np.squeeze(sc_gain, axis=-1)

        # Data driven map contains nan, will be set to 0 here
        sc_gain[np.isnan(sc_gain)] = 0

        return sc_gain

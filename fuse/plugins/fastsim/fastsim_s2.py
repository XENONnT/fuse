import strax
import numpy as np
import straxen
import logging

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.fastsim.fastsim_s2")


@export
class S2Areas(FuseBasePlugin):
    """Plugin to simulate S2 areas from electrons extracted for fastsim

    """
    __version__ = "0.0.1"

    depends_on = ("fastsim_macro_clusters",)
    provides = "fastsim_s2"
    data_kind = "fastsim_events"

    dtype = [(("S2 area, uncorrected [PE]", "s2_area"), np.float32),
             (("Alternate S2 area, uncorrected [PE]", "alt_s2_area"), np.float32),
             (("Sum of S2 areas in event, uncorrected [PE]", "s2_sum"), np.float32),
             (("Number of S2s in event", "multiplicity"), np.int32)
             ] + strax.time_fields

    save_when = strax.SaveWhen.ALWAYS

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

    p_double_pe_emision = straxen.URLConfig(
        default="take://resource://"
                "SIMULATION_CONFIG_FILE.json?&fmt=json"
                "&take=p_double_pe_emision",
        type=(int, float),
        cache=True,
        help="Probability of double photo-electron emission",
    )

    def compute(self, fastsim_macro_clusters):
        eventids = np.unique(fastsim_macro_clusters['eventid'])
        result = np.zeros(len(eventids), dtype=self.dtype)
        result["time"] = fastsim_macro_clusters["time"]
        result["endtime"] = fastsim_macro_clusters["endtime"]
        for i, eventid in enumerate(eventids):
            clusters = fastsim_macro_clusters[fastsim_macro_clusters["eventid"] == eventid]
            s2_areas = []
            for cluster in clusters:
                pos = np.array([cluster["x"], cluster["y"]]).T  # TODO: check if correct positions
                ly = self.get_s2_light_yield(pos)[0]
                area = ly * cluster['n_electron_extracted']
                if area > 0:
                    s2_areas.append(area)
            if len(s2_areas):
                result[i]["s2_area"] = max(s2_areas)
                result[i]["s2_sum"] = sum(s2_areas)
            if len(s2_areas) > 1:
                result[i]["alt_s2_area"] = sorted(s2_areas, reverse=True)[1]
            result[i]["multiplicity"] = len(s2_areas)
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

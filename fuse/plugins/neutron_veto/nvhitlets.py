import strax
import straxen
import numpy as np
import scipy.constants as const
from sklearn.cluster import DBSCAN


from ...plugin import FuseBasePlugin

from ...dtypes import neutron_veto_hitlet_dtype

export, __all__ = strax.exporter()


@export
class NeutronVetoHitlets(FuseBasePlugin):
    """Plugin to simulate the Neutron Veto to hitlets.

    @Experts: Please add a better description of the plugin here.
    """

    __version__ = "0.0.1"

    depends_on = "nv_pmthits"
    provides = "nv_hitlets"
    data_kind = "nv_hitlets"

    dtype = neutron_veto_hitlet_dtype + strax.interval_dtype

    # Fix these URL configs!
    nveto_pmt_qe = straxen.URLConfig(
        default="nveto_pmt_qe://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=nveto_pmt_qe"
        "&fmt=json",
        help="Quantum efficiency of NV PMTs",
    )

    nveto_spe_parameters = straxen.URLConfig(
        default="nveto_spe://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?&key=nveto_pmt_spe&fmt=json",
        help="SPE model of NV PMTs",
    )

    stack_hitlets = straxen.URLConfig(
        default=False,
        help="Option to enable or disable hitlet stacking",
    )

    ce_scaling = straxen.URLConfig(
        type=(int, float),
        default="take://resource://SIMULATION_CONFIG_FILE.json?fmt=json&take=nveto_eCE",
        cache=True,
        help="Add good description here",
    )

    def compute(self, nv_pmthits):

        if len(nv_pmthits) == 0:
            return np.zeros(0, self.dtype)

        hitlets = self._nv_hitlets(nv_pmthits)

        result = np.zeros(len(hitlets), dtype=self.dtype)
        result["time"] = hitlets["time"]
        result["length"] = 1
        result["dt"] = 10
        result["channel"] = hitlets["pmthitID"]
        result["area"] = hitlets["pe_area"]

        return strax.sort_by_time(result)

    def _nv_hitlets(self, pmthits):

        extra_fields = [
            ("pe_area", "<f4"),
            ("cluster_times_ns", "<f4"),
            ("labels", "<i4"),
            ("n_clusters_hits", "<i4"),
        ]
        new_dtype = pmthits.dtype.descr + extra_fields
        pmthits_extended = np.zeros(pmthits.shape, dtype=new_dtype)
        for field in pmthits.dtype.names:
            pmthits_extended[field] = pmthits[field]
        for field, dtype in extra_fields:
            pmthits_extended[field] = np.zeros(pmthits_extended.shape, dtype=dtype)
        pmthits = pmthits_extended

        # select NV PMTs (need to exclude MV PMTs?)
        mask = (pmthits["pmthitID"] >= 2000) & (pmthits["pmthitID"] < 2121)
        pmthits = pmthits[mask]

        self.log.debug("Applying QE")
        # Applying Quantum efficiency for each pmt
        # --- super super slow.....
        # qe = 1e-2 * np.vectorize(self.QE_E)(pmthits["pmthitEnergy"], pmthits["pmthitID"])

        # A faster approach
        qe = np.zeros(len(pmthits), dtype=np.float32)
        unique_ids = np.unique(pmthits["pmthitID"])
        NVeto_PMT_QE = self.nveto_pmt_qe
        for uid in unique_ids:
            mask_id = pmthits["pmthitID"] == uid
            qe[mask_id] = 1e-2 * QE_E(pmthits["pmthitEnergy"][mask_id], uid, NVeto_PMT_QE)

        self.log.debug("Applying CE")
        # Applying effective collection efficiency
        qe *= self.ce_scaling

        # Applying acceptance per pmt: for the approach in which SPE PDF has already applied a threshold for low charges
        self.log.debug("Applying per pmt acceptance")
        # also very slow, think this is a bottleneck of the URLConfigs, it is not very efficient to call them in a loop many times.
        # qe = qe * np.vectorize(self.get_acceptance)(pmthits["pmthitID"])

        NV_SPE = self.nveto_spe_parameters
        acceptance_dict = {k: v["acceptance"] for k, v in NV_SPE.items()}
        qe = qe * np.vectorize(acceptance_dict.get)(pmthits["pmthitID"])

        self.log.debug("Binomial sampling")
        # Generate a photoelectron based on (binomial) conversion probability qe*eCE*spe_acc
        pe = np.array([np.random.binomial(1, j, 1)[0] for j in qe])

        maks_qe = pe > 0
        pmthits = pmthits[maks_qe]

        # 2. Sampling charge from SPE for each pmthit with a generated pe
        self.log.debug("Sampling hitlets charge pe")
        # Same performance problems as above. Lets try something faster
        # pmthits["pe_area"] = np.vectorize(self.pe_charge_N)(pmthits["pmthitID"])
        spe_charge = np.zeros(len(pmthits), dtype=np.float32)
        unique_ids = np.unique(pmthits["pmthitID"])
        for uid in unique_ids:
            mask_id = pmthits["pmthitID"] == uid

            # Just call the random choice once per PMT ID
            SPE_channel = NV_SPE.get(uid)
            spe_charge[mask_id] = self.rng.choice(
                SPE_channel["pe"], p=SPE_channel["SPE_values"], size=np.sum(mask_id)
            )
        pmthits["pe_area"] = spe_charge

        # 3. Creating hitlet times
        self.log.debug("Getting time hitlets")
        times = []
        for i in np.unique(pmthits["evtid"]):
            mask = pmthits["evtid"] == i
            pmthits_evt = pmthits[mask]
            cluster_times_ns = pmthits_evt["pmthitTime"] - min(pmthits_evt["pmthitTime"])
            times.append(cluster_times_ns)
        pmthits["cluster_times_ns"] = np.concatenate(times)

        if not self.stack_hitlets:
            return pmthits

        elif self.stack_hitlets:
            self.log.debug("Looking for stacked hitlets")

            arr_c_evt = []
            for i in np.unique(pmthits["evtid"]):
                arr_evt = pmthits[pmthits["evtid"] == i]
                arr_c_pmt = []
                for j in np.unique(arr_evt["pmthitID"]):
                    arr_pmt = arr_evt[arr_evt["pmthitID"] == j]
                    labels = channel_cluster_nv(arr_pmt["cluster_times_ns"])
                    arr_pmt["labels"] = labels
                    arr_c = np.concatenate(
                        [
                            get_clusters_arrays(arr_pmt[arr_pmt["labels"] == l], new_dtype)
                            for l in np.unique(labels)
                        ]
                    )

                    arr_c_pmt.append(arr_c)
                arr_c_evt.append(np.concatenate(arr_c_pmt))

            return np.concatenate(arr_c_evt)

    # Get Quantum efficiency

    # def QE_E(self, E, ID):
    #     WL = energy_to_wavelenght(E)
    #     ind = ID - 2000
    #     qe = self.nveto_pmt_qe[ind](WL)
    #     # qe = self.nveto_pmt_qe["QE"][ind](WL)

    #     return qe

    # def get_acceptance(self, ID):
    #     acc = self.nveto_spe_parameters.get(ID)["acceptance"]
    #     return acc

    # Get acceptance threshold
    def get_threshold_acc(self, ID):
        ind = ID - 2000
        threshold = self.nveto_spe_parameters.threshold_pe.values[ind]
        return threshold

    # Sampling charge from SPE
    # def pe_charge_N(self, pmt_id):
    #     SPE_channel = self.nveto_spe_parameters.get(pmt_id)

    #     charge = self.rng.choice(SPE_channel["pe"], SPE_channel["SPE_values"], k=1)[0]

    #     return charge


def energy_to_wavelenght(E):
    Joules_to_eV = 1.602 * 1e-19
    return 1e9 * const.h * const.c / (E * Joules_to_eV)


def QE_E(E, ID, nveto_pmt_qe):
    WL = energy_to_wavelenght(E)
    ind = ID - 2000
    qe = nveto_pmt_qe[ind](WL)
    # qe = self.nveto_pmt_qe["QE"][ind](WL)
    return qe


# Cluster for stacket hitlets
def channel_cluster_nv(t):
    db_cluster = DBSCAN(
        eps=8, min_samples=1
    )  # As a preliminar value we fix distance between two photons arriving in the same pmt 8ns
    t_val = np.array(t)
    clusters = np.array(db_cluster.fit_predict(t_val.reshape(-1, 1)))
    return clusters


def get_clusters_arrays(arr, typ):
    arr_nv_c = np.zeros(1, dtype=typ)
    arr_nv_c["n_clusters_hits"] = len(arr)

    for i in arr.dtype.names:  # <-- CORRETTO: usare dtype.names invece di fields
        if i in ["time", "pmthitTime", "cluster_times_ns"]:
            arr_nv_c[i] = np.min(arr[i])
        elif i == "endtime":
            arr_nv_c[i] = np.max(arr[i])
        elif i in ["pe_area", "pmthitEnergy"]:
            arr_nv_c[i] = np.sum(arr[i])
        elif i in ["evtid", "pmthitID", "labels"]:
            arr_nv_c[i] = np.unique(arr[i])

    return arr_nv_c

import strax
import straxen
import numpy as np
import scipy.constants as const
from sklearn.cluster import DBSCAN


from ...plugin import FuseBasePlugin

from ...dtypes import (
    neutron_veto_hitlet_dtype
)

class NeutronVetoHitlets(FuseBasePlugin):
    """
    Plugin to simulate the Neutron Veto to hitlets. 
    
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
        "SIMULATION_CONFIG_FILE.json?&key=nveto_pmt_qe",
        help="Quantum efficiency of NV PMTs",
    )

    # Rename this to be not SR dependent
    nveto_spe_parameters = straxen.URLConfig(
        default="nveto_spe_sr1://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?&key=nveto_spe_sr1",
        help="SR1 SPE model of NV PMTs",
    )

    # Add a few extra configs to remove them elsewhere

    # @experts: would the stacking change the output dtype of the plugin? If yes, it will not work. 
    stack_hitlets = straxen.URLConfig(
        default=False,
        help="Option to enable or disable hitlet stacking",
    )

    # Is this value something you would like to track in the config files or should it just be set here?
    ce_scaling = straxen.URLConfig(
        type=(int, float),
        default=0.75,
        help="Add good description here",
    )

    # Next steps: remove this part. 
    # def __init__(self, sr=0):
    #     self.path = "/home/digangi/private_nt_aux_files/sim_files/"  # pietro - need to modify this to work with urlconfig
    #     self.QE_value = QE_nVeto(self.path + "nveto_pmt_qe.json")
    #     self.SPE_nVeto = SPE_parameters(self.path+'nveto_spe_sr'+str(sr)+'.json')

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

        # Henning comment: I would remove this long comment block below. If it is strictly needed, maybe it can got into the docstring of the class?

        # -------------------------------------------------Arguments---------------------------------------------------#
        # QE_Scaling corrrespond to collection efficiency, no study has been done on the CE of muon Veto we use a default value close to the nVeto see https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:mancuso:hitletsimulator:collection_efficiency
        # period : this could be related to the rate of a source, or the rate for a time bin if we reconstruct an spectrum. If no source 1 second is the default value (see Comments)

        # ----------------------------------------------Commments-----------------------------------------------------------------#:
        # 1.There is no application of a threshold per channel based on the acceptation by default, but we keep the value in the data frame for each pmt, and one can do manually. This is in order to not condition the sampling, and compare it with the data with different cuts.
        # .2. The period is set by default at 1s to care about no pyle up or merge of hitlets if one want to do an analysis for rare events (independent non sourced ones). If we simulate a calibration or a constant flux this value has to be changed to real rate one.

        # 0.---------------Load GEANT output-------------------#

        # Adding extra fields to pmthits np.array
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

        # 1. First step PHOTON to first dinode
        self.log.debug("Applying QE and CE")
        # Applying Quantum efficiency for each pmt
        qe = 1e-2 * np.vectorize(self.QE_E)(pmthits["pmthitEnergy"], pmthits["pmthitID"])
        # Applying effective collection efficiency
        qe *= self.ce_scaling
        # Applying acceptance per pmt: for the approach in which SPE PDF has already applied a threshold for low charges
        qe = qe * np.vectorize(self.get_acceptance)(pmthits["pmthitID"])
        # Generate a photoelectron based on (binomial) conversion probability qe*eCE*spe_acc
        pe = np.array([np.random.binomial(1, j, 1)[0] for j in qe])
        # Discard pmthits which do not generate a pe
        self.log.debug("Loading hit survive")
        maks_qe = pe > 0
        pmthits = pmthits[maks_qe]

        # 2. Sampling charge from SPE for each pmthit with a generated pe
        self.log.debug("Sampling hitlets charge pe")
        pmthits["pe_area"] = np.vectorize(self.pe_charge_N)(pmthits["pmthitID"])

        # 3. Creating hitlet times
        self.log.debug("Getting time hitlets")
        times = []
        for i in np.unique(pmthits["evtid"]):
            mask = pmthits["evtid"] == i
            pmthits_evt = pmthits[mask]
            cluster_times_ns = pmthits_evt["pmthitTime"] - min(pmthits_evt["pmthitTime"])
            times.append(cluster_times_ns)
        pmthits["cluster_times_ns"] = np.concatenate(times)


        # Same comment as above: If this option produces different output dtypes it will not work. One could add a second output to the plugin if needed or add a new plugin that takes pmthits as input and produces the stacked hitlets as output.

        if not self.stack_hitlets:
            return pmthits
        
        # 3.1 Stacked hitlets: this correspond to hitlets in the same pmt with a time difference below some estimated time response of the Channel (8 ns, i.e. 4 samples).
        elif self.stack_hitlets:
            self.log.debug("Looking for stacked hitlets")
            # Here we set times related to the first hit, we only use that for stacket hitlets
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
                    # arr_c =np.concatenate([get_clusters_arrays(arr_pmt[arr_pmt['labels']==l],dtypes) for l in np.unique(labels)])
                    arr_c_pmt.append(arr_c)
                arr_c_evt.append(np.concatenate(arr_c_pmt))

            return np.concatenate(arr_c_evt)

    # Get Quantum efficiency
    def QE_E(self, E, ID):
        WL = energy_to_wavelenght(E)
        ind = ID - 2000
        qe = self.nveto_pmt_qe["QE"][ind](WL)
        return qe

    def get_acceptance(self, ID):
        acc = self.nveto_spe_parameters.get(ID)['acceptance']
        return acc

    # Get acceptance threshold
    def get_threshold_acc(self, ID):
        ind = ID - 2000
        threshold = self.nveto_spe_parameters.threshold_pe.values[ind]
        return threshold

    # Sampling charge from SPE
    def pe_charge_N(self, pmt_id):
        SPE_channel = self.nveto_spe_parameters.get(pmt_id)


        # We can not use the line below as we have to make sure fuse is producing reproducible results. For this reason we have to stick to the random generator of the plugin.
        # charge=rd.choices(SPE_channel['pe'],SPE_channel['SPE_values'],k=1)[0]

        #I'm not sure if numpy choice is exactly the same as random.choices. Please check this.
        charge=self.rng.choice(SPE_channel['pe'],SPE_channel['SPE_values'],k=1)[0]

        return charge


def energy_to_wavelenght(E):
    Joules_to_eV = 1.602 * 1e-19
    return 1e9 * const.h * const.c / (E * Joules_to_eV)

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


# I do not see this used anywhere: -> Delete?
# def create_SPE_file(path, sr="0"):
#     # path to the aux_files
#     # SR : to change configuration for each SR
#     spe_df = pd.DataFrame(columns=["pmtID", "pe", "SPE_values", "acceptance"])
#     array_x = np.load(path + "x_data_sr" + sr + ".npy")
#     array_y = np.load(path + "sr" + sr + "_pdfs.npy")
#     spe_df["pmtID"] = np.arange(2000, 2120)
#     spe_df["pe"] = array_x.tolist()
#     spe_df["SPE_values"] = array_y.tolist()
#     spe_df["acceptance"] = np.load(path + "spe_acc_sr" + sr + ".npy", allow_pickle=True)
#     return np.save(path + "SPE_SR" + sr + ".npy", spe_df.to_records())


# This one is the same as the nveto_spe_sr1_dict function right?
# SPE parameters: ID, pe, SPE, acceptance
# def SPE_parameters(file_spe_model):
#     with open(file_spe_model, 'r') as f:     
#         data_spe = json.load(f) 
#     data_dict = {entry['pmtID']: entry for entry in data_spe}
#     # SPE_ch= pd.DataFrame(columns=['pmtID','pe','SPE','acceptance'])
#     # SPE_ch['pmtID'],SPE_ch['pe'], SPE_ch['SPE'],SPE_ch['acceptance']=data_spe['pmtID'],data_spe['charge'],data_spe['SPE_values'],data_spe['acceptance']
#     # acceptance_ch= [threshold_acc(SPE_ch,i) for i in np.arange(2000,2120)]
#     # SPE_ch['threshold_pe']=acceptance_ch

#     return data_dict

# I do not see this used anywhere: -> Delete?
# def threshold_acc(SPE_df, ID):
#     SPE_ID = pd.DataFrame()
#     SPE_ID["cumulative"] = np.cumsum(SPE_df[SPE_df.pmtID == ID].SPE.values[0])
#     SPE_ID["charges"] = SPE_df[SPE_df.pmtID == ID].pe.values[0]
#     accep = SPE_df[SPE_df.pmtID == ID].acceptance.values[0]
#     threshold = min(SPE_ID[SPE_ID.cumulative >= (1 - accep)].charges.values)
#     return threshold


# This one is the one in the URLConfig nveto_pmt_qe right?
# Quantum efficiency
# def QE_nVeto(Q_E_nveto_file):
#     with open(Q_E_nveto_file, "r") as f:
#         data = json.loads(f.read())
#     QE_array_n = []
#     # nVeto
#     for i in np.arange(2000, 2120):
#         QE_array_n.append(
#             interpolate.interp1d(
#                 data["nv_pmt_qe_wavelength"],
#                 data["nv_pmt_qe"][str(i)],
#                 bounds_error=False,
#                 fill_value=0,
#             )
#         )
#     # Watertank_QE
#     pmt_id = list(np.arange(2000, 2120))
#     QE_array = QE_array_n
#     pd_dict = {"pmt_id": pmt_id, "QE": QE_array}
#     return pd_dict
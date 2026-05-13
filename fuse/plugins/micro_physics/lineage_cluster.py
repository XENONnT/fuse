import numpy as np
import numba
import strax
import straxen

import re
import periodictable as pt

from ...common import stable_argsort
from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

NEST_BETA = (8, 0, 0)
NEST_GAMMA = (7, 0, 0)
NEST_ALPHA = (6, 4, 2)
NEST_NR = (0, 0, 0)
NEST_NONE = (12, 0, 0)

# --- Int-coded string fields -------------------------------------------------
# The hot path inside `build_lineage_for_event` previously did string compares
# on `type`, `parenttype`, `creaproc`, `edproc` for every interaction. Here we
# precompute an int8 code per row once at the top of `compute()`. Unknown
# strings encode as -1 so they never match any named constant.

_TYPE_NAMES = ("alpha", "e-", "e+", "gamma", "neutron")
_PARENTTYPE_NAMES = ("", "none", "neutron", "gamma")
_CREAPROC_NAMES = (
    "RadioactiveDecayBase", "compt", "conv", "phot",
    "photonNuclear", "eBrem", "Transportation",
)
_EDPROC_NAMES = (
    "RadioactiveDecayBase", "compt", "conv", "phot",
    "Transportation", "hadElastic", "neutronInelastic", "nCapture", "ionIoni",
)

_TYPE_CODE = {name: i for i, name in enumerate(_TYPE_NAMES)}
_PARENTTYPE_CODE = {name: i for i, name in enumerate(_PARENTTYPE_NAMES)}
_CREAPROC_CODE = {name: i for i, name in enumerate(_CREAPROC_NAMES)}
_EDPROC_CODE = {name: i for i, name in enumerate(_EDPROC_NAMES)}

# Named constants used by the coded helpers (`is_lineage_broken_coded`,
# `classify_lineage_coded`). Reading these is a single integer dereference.
TYPE_ALPHA = _TYPE_CODE["alpha"]
TYPE_EM = _TYPE_CODE["e-"]
TYPE_EP = _TYPE_CODE["e+"]
TYPE_GAMMA = _TYPE_CODE["gamma"]
TYPE_NEUTRON = _TYPE_CODE["neutron"]

PARENTTYPE_EMPTY = _PARENTTYPE_CODE[""]
PARENTTYPE_NONE = _PARENTTYPE_CODE["none"]
PARENTTYPE_NEUTRON = _PARENTTYPE_CODE["neutron"]
PARENTTYPE_GAMMA = _PARENTTYPE_CODE["gamma"]

CREA_RDB = _CREAPROC_CODE["RadioactiveDecayBase"]
CREA_COMPT = _CREAPROC_CODE["compt"]
CREA_CONV = _CREAPROC_CODE["conv"]
CREA_PHOT = _CREAPROC_CODE["phot"]
CREA_PHOTONNUCLEAR = _CREAPROC_CODE["photonNuclear"]
CREA_EBREM = _CREAPROC_CODE["eBrem"]

EDPROC_RDB = _EDPROC_CODE["RadioactiveDecayBase"]
EDPROC_COMPT = _EDPROC_CODE["compt"]
EDPROC_CONV = _EDPROC_CODE["conv"]
EDPROC_PHOT = _EDPROC_CODE["phot"]
EDPROC_TRANSPORTATION = _EDPROC_CODE["Transportation"]
EDPROC_HADELASTIC = _EDPROC_CODE["hadElastic"]
EDPROC_NEUTRONINELASTIC = _EDPROC_CODE["neutronInelastic"]
EDPROC_NCAPTURE = _EDPROC_CODE["nCapture"]
EDPROC_NEUTRON_BREAK = frozenset((
    EDPROC_TRANSPORTATION, EDPROC_HADELASTIC, EDPROC_NEUTRONINELASTIC, EDPROC_NCAPTURE,
))
PARENTTYPE_NEUTRON_PRIMARY = frozenset((PARENTTYPE_EMPTY, PARENTTYPE_NONE, PARENTTYPE_NEUTRON))


@export
class LineageClustering(FuseBasePlugin):
    """Plugin to cluster interactions based the lineage of the particle track.

    This plugin groups energy deposits into clusters based on a set of
    rules defining if the energy deposits belong to the same lineage or
    not. A lineage is then classified based on the type of the particle
    and its parent.
    """

    __version__ = "0.0.6"

    depends_on = "geant4_interactions"

    provides = "interaction_lineage"

    dtype = [
        (("Lineage index of the energy deposit", "lineage_index"), np.int32),
        (("Event lineage index", "event_lineage_index"), np.int32),
        (("Geant4 lineage track ID", "lineage_trackid"), np.int16),
        (("NEST interaction type", "lineage_type"), np.int32),
        (("Mass number of the interacting particle", "A"), np.int16),
        (("Charge number of the interacting particle", "Z"), np.int16),
        (("Type of the main cluster (alpha beta gamma)", "main_cluster_type"), np.dtype("U10")),
    ] + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    lineages_build = 1

    # Config options
    gamma_distance_threshold = straxen.URLConfig(
        default=0.0,
        type=(int, float),
        help="Distance threshold to break lineage for gamma rays [cm]. \
        Do not break if distance is smaller than threshold. \
        Default at 0 means we always break the lineage.",
    )

    brem_distance_threshold = straxen.URLConfig(
        default=0,
        type=(int, float),
        help="Distance threshold to break lineage for bremsstrahlung [cm]. \
        Do not break if distance is smaller than threshold.",
    )

    time_threshold = straxen.URLConfig(
        default=10,
        type=(int, float),
        help="Time threshold to break the lineage [ns]",
    )

    classify_ic_as_gamma = straxen.URLConfig(
        default=True,
        type=bool,
        help="Classify internal conversion electrons as gamma particles",
    )

    classify_phot_as_beta = straxen.URLConfig(
        default=True,
        type=bool,
        help="Classify photoabsorption electrons as beta particles \
        (if False, classify as gamma particles)",
    )

    def compute(self, geant4_interactions):
        """
        Args:
            geant4_interactions (np.ndarray): An array of GEANT4 interaction data.

        Returns:
            np.ndarray: An array of cluster IDs with corresponding time and endtime values.
        """

        self.log.debug(f"Building lineages for {len(geant4_interactions)} interactions")

        if len(geant4_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        event_id_sort = stable_argsort(geant4_interactions[["eventid", "time", "t"]])
        undo_sort_index = stable_argsort(event_id_sort)
        interactions = geant4_interactions[event_id_sort]

        # Pre-compute int-coded fields once on the full input, then permute
        # the same way as `interactions`. Per-event slices below stay aligned.
        codes = slice_codes(build_codes(geant4_interactions), event_id_sort)

        lineage_ids, lineage_trackids, lineage_types, lineage_A, lineage_Z, main_cluster_type = (
            self.build_lineages(interactions, codes)
        )

        # The lineage index is now unique per event. We need to make it unique for the whole run
        _, unique_lineage_index = np.unique(
            (interactions["eventid"], lineage_ids), axis=1, return_inverse=True
        )

        data = np.zeros(len(interactions), dtype=self.dtype)
        data["lineage_index"] = unique_lineage_index + self.lineages_build
        data["event_lineage_index"] = lineage_ids
        data["lineage_trackid"] = lineage_trackids
        data["lineage_type"] = lineage_types
        data["A"] = lineage_A
        data["Z"] = lineage_Z
        data["main_cluster_type"] = main_cluster_type

        data["time"] = interactions["time"]
        data["endtime"] = interactions["endtime"]

        self.lineages_build = np.max(data["lineage_index"]) + 1

        return data[undo_sort_index]

    def build_lineages(
        self,
        geant4_interactions,
        codes,
    ):

        event_ids = np.unique(geant4_interactions["eventid"])

        all_lineage_ids = []
        all_lineage_trackids = []
        all_lineage_types = []
        all_lineage_As = []
        all_lineage_Zs = []
        all_main_cluster_types = []

        for event_id in event_ids:

            event_mask = geant4_interactions["eventid"] == event_id
            event = geant4_interactions[event_mask]
            event_codes = slice_codes(codes, event_mask)

            track_id_sort = stable_argsort(event[["trackid", "t"]])
            undo_sort_index = stable_argsort(track_id_sort)
            event = event[track_id_sort]
            event_codes = slice_codes(event_codes, track_id_sort)

            lineage = self.build_lineage_for_event(
                event,
                event_codes,
                self.gamma_distance_threshold,
                self.brem_distance_threshold,
                self.time_threshold,
                self.classify_ic_as_gamma,
                self.classify_phot_as_beta,
            )[undo_sort_index]

            all_lineage_ids.append(lineage["lineage_index"])
            all_lineage_trackids.append(lineage["lineage_trackid"])
            all_lineage_types.append(lineage["lineage_type"])
            all_lineage_As.append(lineage["lineage_A"])
            all_lineage_Zs.append(lineage["lineage_Z"])
            all_main_cluster_types.append(lineage["main_cluster_type"])

        return (
            np.concatenate(all_lineage_ids),
            np.concatenate(all_lineage_trackids),
            np.concatenate(all_lineage_types),
            np.concatenate(all_lineage_As),
            np.concatenate(all_lineage_Zs),
            np.concatenate(all_main_cluster_types),
        )

    @staticmethod
    def build_lineage_for_event(
        event,
        codes,
        gamma_distance_threshold,
        brem_distance_threshold,
        time_threshold,
        classify_ic_as_gamma,
        classify_phot_as_beta,
    ):
        """Build the per-interaction lineage assignment for a single event.

        Thin wrapper around the numba kernel: builds CSR-style lookup tables
        and hands the per-row arrays to `_build_lineage_for_event_kernel`.
        `main_cluster_type` stays on the numpy path because numba handles
        structured-string outputs poorly and it is not in the inner-loop hot
        path.
        """

        tmp_dtype = [
            ("lineage_index", np.int32),
            ("lineage_trackid", np.int16),
            ("lineage_type", np.int32),
            ("lineage_A", np.int16),
            ("lineage_Z", np.int16),
            ("main_cluster_type", np.dtype("U10")),
        ]

        n = len(event)
        if n == 0:
            return np.zeros(0, dtype=tmp_dtype)

        main_cluster_type = assign_main_cluster_type_to_event(event)

        trackid = event["trackid"].astype(np.int32, copy=False)
        parentid = event["parentid"].astype(np.int32, copy=False)
        unique_tid, offsets, indices, positions = _build_trackid_csr(trackid)
        parent_pos = _build_parent_pos(parentid, unique_tid)

        li, lt, ltype, lA, lZ = _build_lineage_for_event_kernel(
            trackid,
            parentid,
            codes["type_code"],
            codes["parenttype_code"],
            codes["creaproc_code"],
            codes["edproc_code"],
            codes["type_has_digit"],
            codes["type_has_bracket"],
            codes["ion_A"],
            codes["ion_Z"],
            event["x"].astype(np.float64, copy=False),
            event["y"].astype(np.float64, copy=False),
            event["z"].astype(np.float64, copy=False),
            event["t"].astype(np.float64, copy=False),
            offsets,
            indices,
            positions,
            parent_pos,
            float(gamma_distance_threshold),
            float(brem_distance_threshold),
            float(time_threshold),
            bool(classify_ic_as_gamma),
            bool(classify_phot_as_beta),
        )

        tmp_result = np.zeros(n, dtype=tmp_dtype)
        tmp_result["lineage_index"] = li
        tmp_result["lineage_trackid"] = lt
        tmp_result["lineage_type"] = ltype
        tmp_result["lineage_A"] = lA
        tmp_result["lineage_Z"] = lZ
        tmp_result["main_cluster_type"] = main_cluster_type

        return tmp_result


def precompute_particle_lookup(event):
    """Precompute a lookup dictionary mapping each trackid to a numpy int64 array
    of row indices in `event`.

    Returning numpy arrays (instead of Python lists) lets downstream callers fancy-index
    `event_interactions[indices]` directly without a per-call `np.asarray` allocation.
    """
    trackids = event["trackid"]
    counts = {}
    for tid in trackids:
        counts[tid] = counts.get(tid, 0) + 1
    buffers = {tid: np.empty(n, dtype=np.int64) for tid, n in counts.items()}
    cursor = {tid: 0 for tid in counts}
    for idx in range(len(trackids)):
        tid = trackids[idx]
        buffers[tid][cursor[tid]] = idx
        cursor[tid] += 1
    return buffers


def get_particle(event_interactions, event_lineage, index, trackid_lookup):
    """Returns the particle at the index and the lineage of all interactions of
    the same particle."""
    event = event_interactions[index]
    particle_indices = trackid_lookup[event["trackid"]]
    return event, event_lineage[particle_indices]


def get_last_particle_interaction(event_interactions, particle_lineage, particle_indices):
    """Returns the last (previous in time) interaction of the particle that is
    in the lineage.

    `particle_indices` is the precomputed `trackid_lookup[particle["trackid"]]` slice,
    so we avoid re-scanning `event_interactions["trackid"] == particle["trackid"]`.
    The nonzero check is made explicit on `lineage_index` (rather than relying on
    numpy's structured-array logical-OR semantics).
    """
    all_particle_interactions = event_interactions[particle_indices]

    index_of_last_interaction = np.nonzero(particle_lineage["lineage_index"])[0][-1]

    return (
        all_particle_interactions[index_of_last_interaction],
        particle_lineage[index_of_last_interaction],
    )


def precompute_parent_lookup(event):
    """Precompute a lookup dictionary for parent relationships."""
    parent_lookup = {}
    for idx, (trackid, parentid) in enumerate(zip(event["trackid"], event["parentid"])):
        parent_lookup[trackid] = parentid
    return parent_lookup


def get_parent(event_interactions, event_lineage, particle, parent_lookup, trackid_lookup):
    """Returns the parent particle and its lineage of the given particle.

    Uses the precomputed `trackid_lookup` (built once per event in
    `precompute_particle_lookup`) to fetch the parent's row indices in O(1)
    instead of re-scanning `event_interactions["trackid"] == parent_id`.

    When multiple parent interactions share the same time tag (e.g. a fast
    gamma that Compton-scatters and photoabsorbs within G4's sub-ns time-tag
    precision), tie-break by spatial proximity to the daughter's creation
    position so the daughter is attributed to the nearest same-time vertex
    rather than whichever step happens to be last in the array.
    """
    parent_id = parent_lookup.get(particle["trackid"], None)
    if parent_id is None:
        return None, None

    parent_indices = trackid_lookup.get(parent_id, None)
    if parent_indices is None or len(parent_indices) == 0:
        return None, None

    parent_interactions = event_interactions[parent_indices]
    parent_lineages = event_lineage[parent_indices]

    parent_interactions_time_cut = parent_interactions["t"] <= particle["t"]

    if np.sum(parent_interactions_time_cut) == 0:
        parent_to_return = np.argmin(abs(parent_interactions["t"] - particle["t"]))
        return parent_interactions[parent_to_return], parent_lineages[parent_to_return]

    # Among parents with t <= particle.t, pick the latest time and then
    # tie-break by spatial proximity if multiple steps share that latest time.
    candidates = parent_interactions[parent_interactions_time_cut]
    candidate_lineages = parent_lineages[parent_interactions_time_cut]
    t_max = candidates["t"].max()
    same_time_mask = candidates["t"] == t_max
    if same_time_mask.sum() == 1:
        idx = int(np.where(same_time_mask)[0][0])
        return candidates[idx], candidate_lineages[idx]

    same_time = candidates[same_time_mask]
    same_time_lineages = candidate_lineages[same_time_mask]
    distances = np.sqrt(
        (same_time["x"] - particle["x"]) ** 2
        + (same_time["y"] - particle["y"]) ** 2
        + (same_time["z"] - particle["z"]) ** 2
    )
    nearest = int(np.argmin(distances))
    return same_time[nearest], same_time_lineages[nearest]


def is_particle_in_lineage(lineage):
    """Function to check if a particle is already in a lineage."""

    # All particles in the lineage have not been added to a lineage yet
    if np.all(lineage["lineage_index"] == 0):
        return False
    else:
        return True


def num_there(s):
    return any(i.isdigit() for i in s)


def classify_lineage(particle_interaction, classify_ic_as_gamma, classify_phot_as_beta):
    """Function to classify a new lineage based on the particle and its parent
    information."""

    def classify_gamma(particle_interaction):
        if particle_interaction["edproc"] == "compt":
            return NEST_BETA
        elif particle_interaction["edproc"] == "conv":
            return NEST_BETA
        elif particle_interaction["edproc"] == "phot":
            # This is gamma photoabsorption. Return gamma
            return NEST_BETA if classify_phot_as_beta else NEST_GAMMA
        else:
            # could be rayleigh scattering or something else. Classify it as gamma...
            return NEST_BETA

    # If [ in type, it is a nucleus excitation, decaying electromagnetically
    # this will become the lineage of internal conversion electrons
    if "[" in particle_interaction["type"]:
        return NEST_GAMMA if classify_ic_as_gamma else NEST_BETA

    # NR interactions
    if (particle_interaction["parenttype"] == "neutron") & (
        num_there(particle_interaction["type"])
    ):
        return NEST_NR

    # Neutron as primary particle
    elif (particle_interaction["parenttype"] in ["", "none", "neutron"]) & (
        particle_interaction["type"] == "neutron"
    ):
        return NEST_NR

    # Interactions following a gamma
    elif particle_interaction["parenttype"] == "gamma":
        if particle_interaction["creaproc"] == "compt":
            return NEST_BETA
        elif particle_interaction["creaproc"] == "conv":
            return NEST_BETA
        elif particle_interaction["creaproc"] == "phot":
            return NEST_BETA if classify_phot_as_beta else NEST_GAMMA
        elif particle_interaction["creaproc"] == "photonNuclear":
            # nuclear recoil after photonuclear interaction
            if num_there(particle_interaction["type"]):
                return NEST_NR
            elif particle_interaction["type"] == "neutron":
                return NEST_NR
            # gamma ray after photonuclear interaction
            elif particle_interaction["type"] == "gamma":
                return classify_gamma(particle_interaction)
            else:
                return NEST_NONE
        else:
            # This case should not happen? Classify it as none type
            return NEST_NONE

    # Electrons or positrons that are not created by a gamma.
    elif (particle_interaction["type"] == "e-") | (particle_interaction["type"] == "e+"):
        return NEST_BETA

    # The gamma case
    elif particle_interaction["type"] == "gamma":
        return classify_gamma(particle_interaction)

    # Primaries and decay products
    elif (particle_interaction["creaproc"] == "RadioactiveDecayBase") or (
        particle_interaction["parenttype"] == "none"
    ):
        # Alpha particles
        if particle_interaction["type"] == "alpha":
            return NEST_ALPHA

        # Ions
        elif num_there(particle_interaction["type"]):
            element_number, mass = get_element_and_mass(particle_interaction["type"])
            return 6, mass, element_number

        else:
            # This case should not happen? Classify it as NONE
            return NEST_NONE

    else:
        # No classification possible. Classify it as nontype
        return NEST_NONE


def is_lineage_broken(
    particle,
    parent,
    gamma_distance_threshold,
    brem_distance_threshold,
    time_threshold,
):
    """Function to check if the lineage is broken."""

    # second step of a decay. We want to split the lineage
    if (
        particle["creaproc"] == "RadioactiveDecayBase"
        and particle["edproc"] == "RadioactiveDecayBase"
    ):
        return True

    # In the nest code: lineage is always broken if the parent is an ion
    # this breaks the lineage for all ions, also for alpha decays (we need it)
    # but if it's via an excited nuclear state, we want to keep the lineage
    if (num_there(parent["type"])) and ("[" not in parent["type"]):
        return True

    # For gamma rays, check the distance between the parent and the particle
    if particle["type"] == "gamma":

        if particle["creaproc"] == "phot" and particle["edproc"] == "phot":
            # We do not want to split a photo absorption into two clusters
            # The second photo absorption (that we see) could be x rays
            return False

        # Break the lineage for these transportation gammas
        # Transportations is a special case. They are not real gammas.
        # They are just used to transport the energy
        # to another volume in the detector (teflon, gas, etc.)
        if parent["edproc"] == "Transportation":
            return True

        particle_position = np.array([particle["x"], particle["y"], particle["z"]])
        parent_position = np.array([parent["x"], parent["y"], parent["z"]])
        distance = np.sqrt(np.sum((parent_position - particle_position) ** 2, axis=0))

        if particle["creaproc"] == "eBrem":
            # we do not want to split a bremsstrahlung into two clusters
            # if the distance is really small, it is most likely the same interaction
            if distance < brem_distance_threshold:
                return False

        if distance > gamma_distance_threshold:
            return True

    # break neutron lineage
    if parent["type"] == "neutron":
        if parent["edproc"] in ["Transportation", "hadElastic", "neutronInelastic", "nCapture"]:
            return True

    # I also want to break the lineage if the interaction happens way after the parent interaction
    time_difference = particle["t"] - parent["t"]

    if time_difference > time_threshold:
        return True

    # Otherwise the lineage is not broken
    return False


def is_lineage_broken_coded(
    p_type_code, p_creaproc_code, p_edproc_code, p_t, p_x, p_y, p_z,
    pa_type_code, pa_edproc_code, pa_t, pa_x, pa_y, pa_z,
    pa_has_digit, pa_has_bracket,
    gamma_distance_threshold, brem_distance_threshold, time_threshold,
):
    """Int-coded twin of `is_lineage_broken`. Same control flow, string
    equality replaced with int compares against the named-constant codes."""
    # Second step of a decay
    if p_creaproc_code == CREA_RDB and p_edproc_code == EDPROC_RDB:
        return True

    # Parent is an ion (has digit) but not in excited state (no bracket)
    if pa_has_digit and not pa_has_bracket:
        return True

    if p_type_code == TYPE_GAMMA:
        if p_creaproc_code == CREA_PHOT and p_edproc_code == EDPROC_PHOT:
            return False
        if pa_edproc_code == EDPROC_TRANSPORTATION:
            return True
        dx = pa_x - p_x
        dy = pa_y - p_y
        dz = pa_z - p_z
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if p_creaproc_code == CREA_EBREM and distance < brem_distance_threshold:
            return False
        if distance > gamma_distance_threshold:
            return True

    if pa_type_code == TYPE_NEUTRON and pa_edproc_code in EDPROC_NEUTRON_BREAK:
        return True

    if (p_t - pa_t) > time_threshold:
        return True

    return False


def _classify_gamma_coded(p_edproc_code, classify_phot_as_beta):
    """Int-coded twin of the inner `classify_gamma`."""
    if p_edproc_code == EDPROC_COMPT:
        return NEST_BETA
    if p_edproc_code == EDPROC_CONV:
        return NEST_BETA
    if p_edproc_code == EDPROC_PHOT:
        return NEST_BETA if classify_phot_as_beta else NEST_GAMMA
    return NEST_BETA


def classify_lineage_coded(
    p_type_code, p_creaproc_code, p_edproc_code, p_parenttype_code,
    p_has_digit, p_has_bracket,
    ion_A, ion_Z,
    classify_ic_as_gamma, classify_phot_as_beta,
):
    """Int-coded twin of `classify_lineage`. Returns the same (lineage_class,
    lineage_A, lineage_Z) tuple. Ion (A, Z) is pre-resolved per unique type
    string by `precompute_ion_AZ`, so the regex never fires in the hot path."""
    # Internal-conversion electrons: nucleus excitation, EM decay
    if p_has_bracket:
        return NEST_GAMMA if classify_ic_as_gamma else NEST_BETA

    # NR interactions following a neutron
    if p_parenttype_code == PARENTTYPE_NEUTRON and p_has_digit:
        return NEST_NR

    # Neutron as primary particle
    if (p_parenttype_code in PARENTTYPE_NEUTRON_PRIMARY) and (p_type_code == TYPE_NEUTRON):
        return NEST_NR

    # Interactions following a gamma
    if p_parenttype_code == PARENTTYPE_GAMMA:
        if p_creaproc_code == CREA_COMPT:
            return NEST_BETA
        if p_creaproc_code == CREA_CONV:
            return NEST_BETA
        if p_creaproc_code == CREA_PHOT:
            return NEST_BETA if classify_phot_as_beta else NEST_GAMMA
        if p_creaproc_code == CREA_PHOTONNUCLEAR:
            if p_has_digit:
                return NEST_NR
            if p_type_code == TYPE_NEUTRON:
                return NEST_NR
            if p_type_code == TYPE_GAMMA:
                return _classify_gamma_coded(p_edproc_code, classify_phot_as_beta)
            return NEST_NONE
        return NEST_NONE

    # Electrons or positrons not from a gamma
    if p_type_code == TYPE_EM or p_type_code == TYPE_EP:
        return NEST_BETA

    # The gamma case
    if p_type_code == TYPE_GAMMA:
        return _classify_gamma_coded(p_edproc_code, classify_phot_as_beta)

    # Primaries and decay products
    if p_creaproc_code == CREA_RDB or p_parenttype_code == PARENTTYPE_NONE:
        if p_type_code == TYPE_ALPHA:
            return NEST_ALPHA
        if p_has_digit:
            # Ion: (lineage_class=6, lineage_A=mass, lineage_Z=element_number)
            return (6, ion_A, ion_Z)
        return NEST_NONE

    # No classification possible
    return NEST_NONE


def get_element_and_mass(particle_type):
    """Function to get the element and the mass number from the particle
    type."""

    pattern_match = re.match(r"([a-z]+)([0-9]+)", particle_type, re.I)

    if pattern_match:
        element, mass = pattern_match.groups()
        mass = int(mass)

        element_number = pt.elements.symbol(element).number

    else:
        element_number = None
        mass = None

    return element_number, mass


def _encode_field(strings, code_map):
    """Convert a numpy array of strings into an int8 array via `code_map`.

    Unknown strings encode as -1. Loops over the small set of known names rather
    than the (potentially large) array of rows, so the cost is O(rows * unique_names)
    of vectorised numpy equality.
    """
    out = np.full(len(strings), -1, dtype=np.int8)
    for name, code in code_map.items():
        out[strings == name] = code
    return out


def _has_digit_vectorised(strings):
    """Per-row 'string contains a digit' check, computed once per unique value."""
    unique, inverse = np.unique(strings, return_inverse=True)
    flags = np.fromiter(
        (any(c.isdigit() for c in s) for s in unique),
        dtype=bool,
        count=len(unique),
    )
    return flags[inverse]


def _has_bracket_vectorised(strings):
    """Per-row '[' check, vectorised through numpy.char."""
    return np.char.find(np.asarray(strings, dtype=str), "[") >= 0


def precompute_ion_AZ(geant4_interactions):
    """For each row, look up A (mass) and Z (element number) if the `type`
    string encodes an ion. Cached per unique type string so the regex +
    periodictable lookup runs O(unique_types) instead of O(rows).

    Non-ion rows get (0, 0), matching the implicit None -> 0 conversion in
    the original code path where `tmp_result[i]["lineage_A"] = None` assigned
    zero into the int16 field.
    """
    types = geant4_interactions["type"]
    unique = np.unique(types)
    A_map = {}
    Z_map = {}
    for s in unique:
        if any(c.isdigit() for c in s):
            element_number, mass = get_element_and_mass(s)
            if element_number is not None and mass is not None:
                A_map[s] = mass
                Z_map[s] = element_number
    ion_A = np.zeros(len(types), dtype=np.int16)
    ion_Z = np.zeros(len(types), dtype=np.int16)
    for s, a in A_map.items():
        mask = types == s
        ion_A[mask] = a
        ion_Z[mask] = Z_map[s]
    return ion_A, ion_Z


def build_codes(geant4_interactions):
    """Build the per-row int8 / bool / int16 arrays consumed by the coded
    hot path in `build_lineage_for_event`.

    Returns a dict keyed by field name; every value is a numpy array aligned
    with `geant4_interactions` rows. Callers slice / permute the values the
    same way they slice the input array (so the alignment is preserved
    through the event sort and per-event track sort).
    """
    types = geant4_interactions["type"]
    parents = geant4_interactions["parenttype"]
    crea = geant4_interactions["creaproc"]
    edpr = geant4_interactions["edproc"]
    ion_A, ion_Z = precompute_ion_AZ(geant4_interactions)
    return {
        "type_code": _encode_field(types, _TYPE_CODE),
        "parenttype_code": _encode_field(parents, _PARENTTYPE_CODE),
        "creaproc_code": _encode_field(crea, _CREAPROC_CODE),
        "edproc_code": _encode_field(edpr, _EDPROC_CODE),
        "type_has_digit": _has_digit_vectorised(types),
        "type_has_bracket": _has_bracket_vectorised(types),
        "ion_A": ion_A,
        "ion_Z": ion_Z,
    }


def slice_codes(codes, indices):
    """Apply a permutation, boolean mask, or index array to every entry in `codes`."""
    return {k: v[indices] for k, v in codes.items()}


# --- Numba kernel for the per-particle loop ----------------------------------
#
# Replaces the Python loop in `build_lineage_for_event` with a single `@njit`
# function. The Python wrapper builds CSR-style lookup tables for the per-event
# trackid groups (numba.typed.Dict is much slower than CSR for this access
# pattern) and hands the kernel only flat numpy arrays.


def _build_trackid_csr(trackid):
    """Build CSR-style lookup tables for an event's `trackid` array.

    Returns four arrays:
      `unique_tid` (int32)    sorted unique trackid values
      `offsets`    (int64)    CSR row pointers; group `k` spans
                              [offsets[k] : offsets[k+1]] in `indices`.
      `indices`    (int64)    flat row indices in event, grouped by trackid
                              and (because the event is pre-sorted by
                              `(trackid, t)`) in time order within each group.
      `positions`  (int64)    per-row index into `unique_tid` so the kernel
                              can map `row -> trackid-group` in O(1).
    """
    n = len(trackid)
    sort_idx = np.argsort(trackid, kind="stable")
    sorted_tid = trackid[sort_idx]
    unique_tid, starts = np.unique(sorted_tid, return_index=True)
    offsets = np.concatenate([starts.astype(np.int64), np.array([n], dtype=np.int64)])
    indices = sort_idx.astype(np.int64)
    positions = np.searchsorted(unique_tid, trackid).astype(np.int64)
    return unique_tid, offsets, indices, positions


def _build_parent_pos(parentid, unique_tid):
    """For each row, return the CSR position of its parent trackid in
    `unique_tid`, or -1 if the parent trackid is not in this event."""
    candidate = np.searchsorted(unique_tid, parentid)
    out = np.full(len(parentid), -1, dtype=np.int64)
    in_range = candidate < len(unique_tid)
    matches = np.zeros(len(parentid), dtype=bool)
    matches[in_range] = unique_tid[candidate[in_range]] == parentid[in_range]
    out[matches] = candidate[matches]
    return out


@numba.njit(cache=True)
def _classify_gamma_njit(p_ed, classify_phot_as_beta):
    """Numba twin of `_classify_gamma_coded`. Returns (lineage_class, A, Z)."""
    if p_ed == EDPROC_COMPT:
        return np.int32(8), np.int16(0), np.int16(0)   # NEST_BETA
    if p_ed == EDPROC_CONV:
        return np.int32(8), np.int16(0), np.int16(0)
    if p_ed == EDPROC_PHOT:
        if classify_phot_as_beta:
            return np.int32(8), np.int16(0), np.int16(0)
        return np.int32(7), np.int16(0), np.int16(0)   # NEST_GAMMA
    return np.int32(8), np.int16(0), np.int16(0)


@numba.njit(cache=True)
def _classify_njit(
    p_type, p_crea, p_ed, p_parenttype,
    p_has_digit, p_has_bracket,
    ion_A, ion_Z,
    classify_ic_as_gamma, classify_phot_as_beta,
):
    """Numba twin of `classify_lineage_coded`. Returns (lineage_class, A, Z)
    as a fixed (int32, int16, int16) tuple."""
    # Internal-conversion electrons (nucleus excitation, EM decay)
    if p_has_bracket:
        if classify_ic_as_gamma:
            return np.int32(7), np.int16(0), np.int16(0)
        return np.int32(8), np.int16(0), np.int16(0)

    # NR interactions following a neutron
    if p_parenttype == PARENTTYPE_NEUTRON and p_has_digit:
        return np.int32(0), np.int16(0), np.int16(0)

    # Neutron as primary particle (parent type empty/none/neutron AND particle is neutron)
    if (p_parenttype == PARENTTYPE_EMPTY or p_parenttype == PARENTTYPE_NONE
        or p_parenttype == PARENTTYPE_NEUTRON) and p_type == TYPE_NEUTRON:
        return np.int32(0), np.int16(0), np.int16(0)

    # Interactions following a gamma
    if p_parenttype == PARENTTYPE_GAMMA:
        if p_crea == CREA_COMPT:
            return np.int32(8), np.int16(0), np.int16(0)
        if p_crea == CREA_CONV:
            return np.int32(8), np.int16(0), np.int16(0)
        if p_crea == CREA_PHOT:
            if classify_phot_as_beta:
                return np.int32(8), np.int16(0), np.int16(0)
            return np.int32(7), np.int16(0), np.int16(0)
        if p_crea == CREA_PHOTONNUCLEAR:
            if p_has_digit:
                return np.int32(0), np.int16(0), np.int16(0)
            if p_type == TYPE_NEUTRON:
                return np.int32(0), np.int16(0), np.int16(0)
            if p_type == TYPE_GAMMA:
                return _classify_gamma_njit(p_ed, classify_phot_as_beta)
            return np.int32(12), np.int16(0), np.int16(0)
        return np.int32(12), np.int16(0), np.int16(0)

    # Electrons or positrons not from a gamma
    if p_type == TYPE_EM or p_type == TYPE_EP:
        return np.int32(8), np.int16(0), np.int16(0)

    # The gamma case
    if p_type == TYPE_GAMMA:
        return _classify_gamma_njit(p_ed, classify_phot_as_beta)

    # Primaries and decay products
    if p_crea == CREA_RDB or p_parenttype == PARENTTYPE_NONE:
        if p_type == TYPE_ALPHA:
            return np.int32(6), np.int16(4), np.int16(2)
        if p_has_digit:
            return np.int32(6), ion_A, ion_Z
        return np.int32(12), np.int16(0), np.int16(0)

    return np.int32(12), np.int16(0), np.int16(0)


@numba.njit(cache=True)
def _is_broken_njit(
    p_type, p_crea, p_ed, p_t, p_x, p_y, p_z,
    pa_type, pa_ed, pa_t, pa_x, pa_y, pa_z,
    pa_has_digit, pa_has_bracket,
    gamma_distance_threshold, brem_distance_threshold, time_threshold,
):
    """Numba twin of `is_lineage_broken_coded`."""
    if p_crea == CREA_RDB and p_ed == EDPROC_RDB:
        return True

    if pa_has_digit and not pa_has_bracket:
        return True

    if p_type == TYPE_GAMMA:
        if p_crea == CREA_PHOT and p_ed == EDPROC_PHOT:
            return False
        if pa_ed == EDPROC_TRANSPORTATION:
            return True
        dx = pa_x - p_x
        dy = pa_y - p_y
        dz = pa_z - p_z
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if p_crea == CREA_EBREM and distance < brem_distance_threshold:
            return False
        if distance > gamma_distance_threshold:
            return True

    if pa_type == TYPE_NEUTRON:
        if (pa_ed == EDPROC_TRANSPORTATION or pa_ed == EDPROC_HADELASTIC
            or pa_ed == EDPROC_NEUTRONINELASTIC or pa_ed == EDPROC_NCAPTURE):
            return True

    if (p_t - pa_t) > time_threshold:
        return True

    return False


@numba.njit(cache=True)
def _build_lineage_for_event_kernel(
    trackid, parentid,
    type_code, parenttype_code, creaproc_code, edproc_code,
    type_has_digit, type_has_bracket,
    ion_A_arr, ion_Z_arr,
    x, y, z, t,
    trackid_offsets, trackid_indices, trackid_pos, parent_pos,
    gamma_distance_threshold, brem_distance_threshold, time_threshold,
    classify_ic_as_gamma, classify_phot_as_beta,
):
    """The numba-compiled body of `build_lineage_for_event`.

    Inputs are flat numpy arrays. The CSR encoding (`trackid_offsets`,
    `trackid_indices`, `trackid_pos`) replaces the Python `trackid_lookup`
    dict. `parent_pos[i]` is the CSR position of row i's parent trackid (-1
    if not in this event), replacing `parent_lookup`. Returns five output
    arrays which the Python wrapper assembles into the structured result.
    """
    n = trackid.shape[0]
    n_unique = trackid_offsets.shape[0] - 1

    lineage_index = np.zeros(n, dtype=np.int32)
    lineage_trackid = np.zeros(n, dtype=np.int16)
    lineage_type = np.zeros(n, dtype=np.int32)
    lineage_A = np.zeros(n, dtype=np.int16)
    lineage_Z = np.zeros(n, dtype=np.int16)

    visited = np.zeros(n_unique, dtype=np.bool_)
    running = np.int32(0)

    for i in range(n):
        tid = trackid[i]
        pos = trackid_pos[i]

        if not visited[pos]:
            visited[pos] = True

            # --- get_parent: find parent's row index ---
            parent_idx = np.int64(-1)
            ppos = parent_pos[i]
            if ppos >= 0:
                p_start = trackid_offsets[ppos]
                p_end = trackid_offsets[ppos + 1]
                t_i = t[i]
                # Inlined parent-finding, matching `get_parent`:
                #   (1) find t_max = max(t[idx] for idx in slice if t[idx] <= t_i)
                #   (2) among entries with t[idx] == t_max, pick the one
                #       spatially closest to (x[i], y[i], z[i]).
                # If no entry has t[idx] <= t_i, fall back to nearest |t - t_i|.
                # Squared distance is enough — argmin is the same as for sqrt.
                t_max = np.float64(0.0)
                has_le = False
                for jj in range(p_start, p_end):
                    idx = trackid_indices[jj]
                    t_idx = t[idx]
                    if t_idx <= t_i:
                        if not has_le or t_idx > t_max:
                            t_max = t_idx
                            has_le = True
                best_idx = np.int64(-1)
                if has_le:
                    best_d2 = np.float64(-1.0)
                    x_i = x[i]
                    y_i = y[i]
                    z_i = z[i]
                    for jj in range(p_start, p_end):
                        idx = trackid_indices[jj]
                        if t[idx] == t_max:
                            dx = x[idx] - x_i
                            dy = y[idx] - y_i
                            dz = z[idx] - z_i
                            d2 = dx * dx + dy * dy + dz * dz
                            if best_d2 < 0 or d2 < best_d2:
                                best_d2 = d2
                                best_idx = idx
                else:
                    # Fall back to nearest |t - t_i|.
                    min_diff = np.float64(-1.0)
                    for jj in range(p_start, p_end):
                        idx = trackid_indices[jj]
                        diff = t[idx] - t_i
                        if diff < 0:
                            diff = -diff
                        if min_diff < 0 or diff < min_diff:
                            min_diff = diff
                            best_idx = idx
                parent_idx = best_idx

            if parent_idx >= 0:
                broken = _is_broken_njit(
                    type_code[i], creaproc_code[i], edproc_code[i],
                    t[i], x[i], y[i], z[i],
                    type_code[parent_idx], edproc_code[parent_idx],
                    t[parent_idx], x[parent_idx], y[parent_idx], z[parent_idx],
                    type_has_digit[parent_idx], type_has_bracket[parent_idx],
                    gamma_distance_threshold, brem_distance_threshold, time_threshold,
                )
                if broken:
                    running += np.int32(1)
                    lt, lA, lZ = _classify_njit(
                        type_code[i], creaproc_code[i], edproc_code[i],
                        parenttype_code[i],
                        type_has_digit[i], type_has_bracket[i],
                        ion_A_arr[i], ion_Z_arr[i],
                        classify_ic_as_gamma, classify_phot_as_beta,
                    )
                    lineage_index[i] = running
                    lineage_trackid[i] = np.int16(tid)
                    lineage_type[i] = lt
                    lineage_A[i] = lA
                    lineage_Z[i] = lZ
                else:
                    lineage_index[i] = lineage_index[parent_idx]
                    lineage_trackid[i] = lineage_trackid[parent_idx]
                    lineage_type[i] = lineage_type[parent_idx]
                    lineage_A[i] = lineage_A[parent_idx]
                    lineage_Z[i] = lineage_Z[parent_idx]
            else:
                # No parent — start a new lineage.
                running += np.int32(1)
                lt, lA, lZ = _classify_njit(
                    type_code[i], creaproc_code[i], edproc_code[i],
                    parenttype_code[i],
                    type_has_digit[i], type_has_bracket[i],
                    ion_A_arr[i], ion_Z_arr[i],
                    classify_ic_as_gamma, classify_phot_as_beta,
                )
                lineage_index[i] = running
                lineage_trackid[i] = np.int16(tid)
                lineage_type[i] = lt
                lineage_A[i] = lA
                lineage_Z[i] = lZ
        else:
            # Already-seen trackid — find this trackid's most recent assigned row
            # (largest idx in the CSR slice with idx < i and lineage_index != 0).
            tid_start = trackid_offsets[pos]
            tid_end = trackid_offsets[pos + 1]
            last_idx = np.int64(-1)
            for jj in range(tid_start, tid_end):
                idx = trackid_indices[jj]
                if idx >= i:
                    break
                if lineage_index[idx] != 0:
                    last_idx = idx

            broken = _is_broken_njit(
                type_code[i], creaproc_code[i], edproc_code[i],
                t[i], x[i], y[i], z[i],
                type_code[last_idx], edproc_code[last_idx],
                t[last_idx], x[last_idx], y[last_idx], z[last_idx],
                type_has_digit[last_idx], type_has_bracket[last_idx],
                gamma_distance_threshold, brem_distance_threshold, time_threshold,
            )
            if broken:
                running += np.int32(1)
                lt, lA, lZ = _classify_njit(
                    type_code[i], creaproc_code[i], edproc_code[i],
                    parenttype_code[i],
                    type_has_digit[i], type_has_bracket[i],
                    ion_A_arr[i], ion_Z_arr[i],
                    classify_ic_as_gamma, classify_phot_as_beta,
                )
                lineage_index[i] = running
                lineage_trackid[i] = np.int16(tid)
                lineage_type[i] = lt
                lineage_A[i] = lA
                lineage_Z[i] = lZ
            else:
                lineage_index[i] = lineage_index[last_idx]
                lineage_trackid[i] = lineage_trackid[last_idx]
                lineage_type[i] = lineage_type[last_idx]
                lineage_A[i] = lineage_A[last_idx]
                lineage_Z[i] = lineage_Z[last_idx]

    return lineage_index, lineage_trackid, lineage_type, lineage_A, lineage_Z


def assign_main_cluster_type_to_event(event):
    # Initialize columns for processing
    tracks = event["trackid"]
    parents = event["parentid"]
    types = event["type"]
    creaproc = event["creaproc"]
    edproc = event["edproc"]

    # Initialize mega cluster types with 'None'
    main_cluster_types = np.full(len(tracks), "None", dtype=object)

    # Directly classify the initial interactions
    is_alpha = (types == "alpha") | (edproc == "ionIoni")
    is_beta = (creaproc == "RadioactiveDecayBase") & (types == "e-")
    is_gamma = (creaproc == "RadioactiveDecayBase") & (types == "gamma")
    is_beta_brem = (creaproc == "eBrem") & (types == "gamma")

    # Apply initial classifications
    main_cluster_types[is_alpha] = "alpha"
    main_cluster_types[is_beta] = "beta"
    main_cluster_types[is_gamma] = "gamma"

    # Function to propagate a single mega type
    def propagate_mega_type(mega_type):

        previous_assigned_tracks = set()

        while True:

            assigned = set(tracks[main_cluster_types == mega_type])

            # If no new tracks have been assigned, break
            if assigned == previous_assigned_tracks:
                break

            # Update previous assigned tracks
            previous_assigned_tracks = assigned

            # Find all tracks that are children of the assigned tracks
            children_mask = np.in1d(parents, list(assigned))

            # Assign the mega type to the children
            main_cluster_types[children_mask] = mega_type

    # Propagate each mega type
    for mega_type in ["alpha", "beta", "gamma"]:
        propagate_mega_type(mega_type)

    # We need to propagate beta_brem separately, because we overwrite the beta type
    main_cluster_types[is_beta_brem] = "beta_brem"
    propagate_mega_type("beta_brem")

    return main_cluster_types


def start_new_lineage(
    particle, tmp_result, i, running_lineage_index, classify_ic_as_gamma, classify_phot_as_beta
):

    lineage_class, lineage_A, lineage_Z = classify_lineage(
        particle, classify_ic_as_gamma, classify_phot_as_beta
    )
    tmp_result[i]["lineage_index"] = running_lineage_index
    tmp_result[i]["lineage_trackid"] = particle["trackid"]
    tmp_result[i]["lineage_type"] = lineage_class
    tmp_result[i]["lineage_A"] = lineage_A
    tmp_result[i]["lineage_Z"] = lineage_Z

    return tmp_result


def continue_lineage(particle, tmp_result, i, parent_lineage):

    tmp_result[i]["lineage_index"] = parent_lineage["lineage_index"]
    tmp_result[i]["lineage_trackid"] = parent_lineage["lineage_trackid"]
    tmp_result[i]["lineage_type"] = parent_lineage["lineage_type"]
    tmp_result[i]["lineage_A"] = parent_lineage["lineage_A"]
    tmp_result[i]["lineage_Z"] = parent_lineage["lineage_Z"]

    return tmp_result

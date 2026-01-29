import straxen
import strax
import numpy as np
import numba

export, __all__ = strax.exporter()


@export
class EnergyCut(strax.CutPlugin):
    """Plugin evaluates if the sum of the events energy is below a
    threshold."""

    depends_on = [
        "clustered_interactions",
    ]

    __version__ = "0.0.3"

    provides = "energy_range_cut"
    cut_name = "energy_range_cut"
    cut_description = "Selects events below a certain energy threshold"

    # Config options
    max_energy = straxen.URLConfig(
        default=500,
        type=(int, float),
        help="Upper limit of the event energy in this simulation",
    )

    min_energy = straxen.URLConfig(
        default=0,
        type=(int, float),
        help="Lower limit of the event energy in this simulation",
    )

    def cut_by(self, clustered_interactions):

        energies = build_energies(clustered_interactions)

        self.log.info(f"Applying energy cut: {self.min_energy} keV < E < {self.max_energy} keV")

        self.log.info(f"Event energies range from {np.min(energies)} keV to {np.max(energies)} keV")

        mask = energies < self.max_energy
        mask = mask & (energies > self.min_energy)

        self.log.info(f"Keeping {np.sum(mask)} out of {len(mask)} events")

        return mask


def build_energies(interactions):

    split_energies, event_ids = group_interaction_energies_by_event_id(
        interactions["ed"], interactions["eventid"]
    )

    energy_per_event = [np.sum(e) for e in split_energies]
    energy_event_mapping = dict(zip(event_ids, energy_per_event))
    energies = np.array([energy_event_mapping[event_id] for event_id in interactions["eventid"]])
    return energies


def group_interaction_energies_by_event_id(energy, event_id):

    sort_index = np.argsort(event_id)

    event_id_sorted = event_id[sort_index]
    energy_sorted = energy[sort_index]

    unique_event_id, split_position = np.unique(event_id_sorted, return_index=True)
    return np.split(energy_sorted, split_position[1:]), unique_event_id


@export
class NRCut(strax.CutPlugin):
    """Plugin which filters the microphysics summary for valid NR events."""

    depends_on = ["clustered_interactions", "quanta"]

    __version__ = "0.0.2"

    provides = "nr_cut"
    cut_name = "nr_cut"
    cut_description = "Selects valid NR events in microphysics summary"

    g1_photon_yield = straxen.URLConfig(
        default=0.1,
        type=(int, float),
        help="Scaled g1 x 0.8 to account for corrections [pe/ph]",
    )
    g2_electron_yield = straxen.URLConfig(
        default=13.4,
        type=(int, float),
        help="Scaled g2 x 0.8 to account for corrections [pe/e]",
    )

    max_s1_area = straxen.URLConfig(
        default=700,
        type=(int, float),
        help="Max S1 area [pe] for NR roi",
    )
    max_s2_area = straxen.URLConfig(
        default=3 * 10**4,
        type=(int, float),
        help="Max S2 area [pe] for NR roi",
    )

    def cut_by(self, clustered_interactions):

        vertex_to_keep = filter_events(
            clustered_interactions,
            self.g1_photon_yield,
            self.g2_electron_yield,
            self.max_s1_area,
            self.max_s2_area,
        )

        mask = vertex_to_keep.astype(bool)

        self.log.info(f"Keeping {np.sum(mask)} out of {len(mask)} interactions after NR cut")

        return mask


@numba.njit
def filter_events(mps, g1, g2, max_s1, max_s2):
    """Small function to filter microphysics for valid NR events.

    We cut all events which are overshadowed by other events outside of
    our ROI excluding delayed deexcitations. To account for missing
    detector corrections one should scale g1/g2 down.
    """

    event_ids = np.unique(mps["eventid"])

    vertex_to_keep = np.ones(len(mps))

    vertex_i = 0

    for event_i in event_ids:
        start_index = vertex_i
        max_photons = 0
        max_electrons = 0
        prompt_photons = 0
        number_of_nr_interactions = 0
        start_time = mps["time"][vertex_i]

        for vertex_i in range(start_index, len(mps)):
            vertex = mps[vertex_i]

            _is_a_new_event = event_i < vertex["eventid"]
            if _is_a_new_event:
                # Next event starts break for loop and check next event
                break

            # Is prompt vertex:
            _is_prompt = (vertex["time"] - start_time) < 200  # ns
            if _is_prompt:
                prompt_photons += vertex["photons"]

            # Ignore and drop vertex if too much delayed within event:
            _vertex_is_delayed = (vertex["time"] - start_time) > 3_000_000  # ns (3 ms)
            if _vertex_is_delayed:
                vertex_to_keep[vertex_i] = 0
                continue

            max_photons = max(max_photons, vertex["photons"])
            max_electrons = max(max_electrons, vertex["electrons"])
            _is_nr = vertex["nestid"] == 0
            if _is_nr:
                number_of_nr_interactions += 1

        # Determine the end index for the current event
        # If we broke, vertex_i points to the next event, so end is vertex_i
        # If we didn't break, vertex_i is the last vertex, so end is vertex_i + 1
        end_index = vertex_i if vertex_i < len(mps) and mps["eventid"][vertex_i] != event_i else vertex_i + 1

        # Check if the largest interaction is still within ROI:
        _is_in_nr_roi = (
            (max_photons * g1 < max_s1)
            & (max_electrons * g2 < max_s2)
            & (number_of_nr_interactions > 0)
            & (prompt_photons * g1 < max_s1)
        )

        if not _is_in_nr_roi:
            vertex_to_keep[start_index:end_index] = 0
    return vertex_to_keep

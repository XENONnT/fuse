import straxen
import strax
import numpy as np


class EnergyCut(strax.CutPlugin):
    """Plugin evaluates if the sum of the events energy is below a
    threshold."""

    depends_on = [
        "clustered_interactions",
        "tpc_selection",
        "below_cathode_selection",
    ]

    __version__ = "0.0.1"

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

        energies = -np.ones(len(clustered_interactions))

        volume_mask = (
            clustered_interactions["tpc_selection"]
            | clustered_interactions["below_cathode_selection"]
        )

        energies[volume_mask] = build_energies(clustered_interactions[volume_mask])

        mask = energies < self.max_energy
        mask = mask & (energies > self.min_energy)

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

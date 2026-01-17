import straxen
import strax
import numpy as np


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

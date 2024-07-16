import pandas as pd
import numpy as np

test_root_file_name = "test_cryo_neutrons_tpc-nveto.root"


def build_random_instructions(n):
    df = pd.DataFrame()

    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    df["x"] = r * np.cos(t)
    df["y"] = r * np.sin(t)
    df["z"] = np.random.uniform(-150, -1, n)

    df["photons"] = np.random.uniform(100, 5000, n)
    df["electrons"] = np.random.uniform(100, 5000, n)
    df["excitons"] = df["photons"]
    df["exciton_to_photon_ratio"] = np.ones(n)  # Lets use a ratio of 1 for the tests.

    df["e_field"] = np.array([23] * n)
    df["nestid"] = np.array([7] * n)
    df["ed"] = np.zeros(n)

    # Just set the time with respect to the start of the event
    # The events will be distributed in time by fuse
    df["t"] = np.zeros(n)

    df["eventid"] = np.arange(n)
    df["cluster_id"] = np.arange(n)

    return df

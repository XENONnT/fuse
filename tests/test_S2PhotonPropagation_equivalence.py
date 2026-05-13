"""Bit-identical equivalence test for `propagated_s2_photons` output.

The optimisation work on `fuse/plugins/detector_physics/s2_photon_propagation.py`
must keep every field of `propagated_s2_photons` identical to a frozen baseline.
This test runs the S2PhotonPropagation chain on the standard tests fixture G4
ROOT and asserts ``np.array_equal`` field-by-field against the baseline saved
at ``tests/data/s2_photon_propagation_baseline.npz``.

The baseline is produced once on the `vanilla` clone (currently main = 1.6.3):

    cd /project/lgrandi/mpierre/fuse-opti/fuse-vanilla
    python -m tests.test_S2PhotonPropagation_equivalence regen

Re-generation is gated: it refuses to overwrite the baseline if
``fuse/plugins/detector_physics/s2_photon_propagation.py`` differs from
origin/main, OR if it has uncommitted changes. Bypass with ``--force``.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import timeout_decorator

import fuse
import utilix

from _utils import test_root_file_name
from _utils import test_simulation_config


TIMEOUT = 600

BASELINE_PATH = Path(__file__).parent / "data" / "s2_photon_propagation_baseline.npz"

FIELDS = (
    "channel",
    "dpe",
    "photon_gain",
    "cluster_id",
    "photon_type",
    "time",
    "endtime",
)


def _make_test_context(work_dir):
    st = fuse.context.xenonnt_fuse_full_chain_simulation(
        output_folder=work_dir,
        clustering_method="lineage",
        simulation_config=test_simulation_config,
    )
    st.set_config(
        {
            "path": work_dir,
            "file_name": test_root_file_name,
        }
    )
    return st


def _download_test_g4(work_dir):
    downloader = utilix.mongo_storage.MongoDownloader(store_files_at=(work_dir,))
    downloader.download_single(test_root_file_name, human_readable_file_name=True)
    if not os.path.exists(os.path.join(work_dir, test_root_file_name)):
        raise RuntimeError(f"failed to download {test_root_file_name} into {work_dir}")


def _compute_s2_photons(work_dir, run_number):
    st = _make_test_context(work_dir)
    st.make(run_number, "propagated_s2_photons")
    return st.get_array(run_number, "propagated_s2_photons")


class TestS2PhotonPropagationEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.run_number = "TestRun_00000"

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        _download_test_g4(self.temp_dir.name)

    def tearDown(self):
        shutil.rmtree(self.temp_dir.name)
        os.makedirs(self.temp_dir.name)

    @timeout_decorator.timeout(TIMEOUT, exception_message="S2PhotonPropagation equivalence timed out")
    def test_equivalence(self):
        if not BASELINE_PATH.exists():
            self.skipTest(
                f"baseline not found at {BASELINE_PATH}; "
                "regenerate with `python -m tests.test_S2PhotonPropagation_equivalence regen`"
            )

        data = _compute_s2_photons(self.temp_dir.name, self.run_number)
        baseline = np.load(BASELINE_PATH)

        missing = [f for f in FIELDS if f not in baseline.files]
        self.assertEqual(
            missing, [], msg=f"baseline npz is missing fields: {missing}"
        )

        self.assertEqual(
            len(data),
            len(baseline[FIELDS[0]]),
            msg=f"row count mismatch: produced {len(data)}, baseline {len(baseline[FIELDS[0]])}",
        )

        for name in FIELDS:
            np.testing.assert_array_equal(
                data[name],
                baseline[name],
                err_msg=f"field '{name}' differs from baseline",
            )


def _regen_guard(force=False):
    repo = Path(__file__).resolve().parents[1]
    plugin_rel = "fuse/plugins/detector_physics/s2_photon_propagation.py"

    if force:
        return

    try:
        diff = subprocess.check_output(
            ["git", "status", "--porcelain", plugin_rel],
            cwd=str(repo),
            stderr=subprocess.STDOUT,
        ).decode().strip()
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"git status failed: {e.output.decode() if e.output else e}; "
            "rerun with --force if you know what you're doing"
        )

    if diff:
        raise SystemExit(
            f"refusing to regenerate baseline: {plugin_rel} has uncommitted changes\n"
            f"  {diff}\n"
            "either commit/stash first, or rerun with --force"
        )

    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{plugin_rel}"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        main_sha = subprocess.check_output(
            ["git", "rev-parse", f"origin/main:{plugin_rel}"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        head_sha = main_sha = None

    if head_sha and main_sha and head_sha != main_sha:
        sys.stderr.write(
            f"warning: {plugin_rel} on HEAD ({head_sha[:8]}) differs from origin/main ({main_sha[:8]}). "
            "Baseline must reflect the vanilla algorithm. "
            "Rerun with --force if you've verified the diff is intentional.\n"
        )
        raise SystemExit(1)


def regen(force=False):
    _regen_guard(force=force)

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as work_dir:
        _download_test_g4(work_dir)
        data = _compute_s2_photons(work_dir, "TestRun_00000")

    payload = {name: data[name] for name in FIELDS}
    np.savez_compressed(BASELINE_PATH, **payload)
    print(f"[regen] wrote baseline ({len(data)} rows) to {BASELINE_PATH}")
    for name in FIELDS:
        col = payload[name]
        print(f"  {name:24s}  dtype={col.dtype!s:14s}  shape={col.shape}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "regen":
        force = "--force" in sys.argv[2:]
        regen(force=force)
    else:
        unittest.main()

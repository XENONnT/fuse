import strax
import numpy as np
from immutabledict import immutabledict
from ..secondary_scintillation import SecondaryScintillation

export, __all__ = strax.exporter()


@export
class DelayedElectronsSecondaryScintillation(SecondaryScintillation):
    """This class is used to simulate the extraction of electrons from the
    sources of electron afterpulses."""

    __version__ = "0.0.1"

    child_plugin = True

    result_name_photons = "delayed_electrons_s2_photons"
    result_name_photons_sum = "delayed_electrons_s2_photons_sum"

    depends_on = (
        "drifted_delayed_electrons",
        "extracted_delayed_electrons",
        "delayed_electrons_time",
        "photo_ionization_electrons",
    )

    provides = (result_name_photons, result_name_photons_sum)
    data_kind = {
        result_name_photons: "delayed_individual_electrons",
        result_name_photons_sum: "delayed_interactions_in_roi",
    }

    dtype_photons = [
        (("Number of photons produced by the extracted electron", "n_s2_photons"), np.int32),
    ] + strax.time_fields
    dtype_sum_photons = [
        (
            (
                "Sum of all photons produced by electrons originating from the same cluster",
                "sum_s2_photons",
            ),
            np.int32,
        ),
    ] + strax.time_fields

    dtype = dict()
    dtype[result_name_photons] = dtype_photons
    dtype[result_name_photons_sum] = dtype_sum_photons

    save_when = immutabledict(
        {result_name_photons: strax.SaveWhen.TARGET, result_name_photons_sum: strax.SaveWhen.ALWAYS}
    )

    def compute(self, delayed_interactions_in_roi, delayed_individual_electrons):
        return super().compute(
            interactions_in_roi=delayed_interactions_in_roi,
            individual_electrons=delayed_individual_electrons,
        )

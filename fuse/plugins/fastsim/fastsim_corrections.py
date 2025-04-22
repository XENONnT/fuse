import strax
import straxen
import logging
import numpy as np

from ...plugin import FuseBasePlugin

export, __all__ = strax.exporter()

logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger("fuse.fastsim.fastsim_s1")


@export
class FastsimCorrections(FuseBasePlugin):
    """Plugin to simulate S1 and (alt) S2 areas from photon hits and electrons
    extracted."""

    __version__ = "0.0.1"

    depends_on = ("fastsim_events_uncorrected",)
    provides = "fastsim_corrections"
    data_kind = "fastsim_events"
    dtype = [
        (("Corrected area of main S1 [PE]", "cs1"), np.float32),
        (("Corrected area of main S2 [PE]", "cs2"), np.float32),
        (("Corrected area of alternate S2 [PE]", "alt_cs2"), np.float32),
        (("Main interaction x-position, field-distortion corrected (cm)", "x"), np.float32),
        (
            (
                "Alternative S2 interaction (rel. main S1) x-position, field-distortion corrected (cm)",
                "alt_s2_x_fdc",
            ),
            np.float32,
        ),
        (("Main interaction y-position, field-distortion corrected (cm)", "y"), np.float32),
        (
            (
                "Alternative S2 interaction (rel. main S1) y-position, field-distortion corrected (cm)",
                "alt_s2_y_fdc",
            ),
            np.float32,
        ),
        (("Main interaction r-position, field-distortion corrected (cm)", "r"), np.float32),
        (
            (
                "Alternative S2 interaction (rel. main S1) r-position, field-distortion corrected (cm)",
                "alt_s2_r_fdc",
            ),
            np.float32,
        ),
        # (('Main interaction z-position, field-distortion corrected (cm)', 'z'), np.float32),
        (
            (
                "Interaction z-position, corrected to non-uniform drift velocity, duplicated (cm)",
                "z_dv_corr",
            ),
            np.float32,
        ),
        (
            (
                "Alternative S2 z-position (rel. main S1), corrected to non-uniform drift velocity (cm)",
                "alt_s2_z",
            ),
            np.float32,
        ),
        (
            (
                "Alternative S2 z-position (rel. main S1), corrected to non-uniform drift velocity, duplicated (cm)",
                "alt_s2_z_dv_corr",
            ),
            np.float32,
        ),
        (
            (
                "Correction added to r_naive for field distortion (cm)",
                "r_field_distortion_correction",
            ),
            np.float32,
        ),
        (
            (
                "Correction added to alt_s2_r_naive for field distortion (cm)",
                "alt_s2_r_field_distortion_correction",
            ),
            np.float32,
        ),
        (("Main interaction angular position (radians)", "theta"), np.float32),
        (
            (
                "	Alternative S2 (rel. main S1) interaction angular position (radians)",
                "alt_s2_theta",
            ),
            np.float32,
        ),
    ] + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    electron_lifetime_liquid = straxen.URLConfig(
        default="take://resource://"
        "SIMULATION_CONFIG_FILE.json?&fmt=json"
        "&take=electron_lifetime_liquid",
        type=(int, float),
        cache=True,
        help="Electron lifetime in liquid xenon [ns]",
    )

    s2_correction_map = straxen.URLConfig(
        default="itp_map://resource://simulation_config://"
        "SIMULATION_CONFIG_FILE.json?"
        "&key=s2_correction_map"
        "&fmt=json",
        cache=True,
        help="S2 correction map",
    )

    fdc_map = straxen.URLConfig(
        infer_type=False,
        help="3D field distortion correction map path",
        default="legacy-fdc://xenon1t_sr0_sr1?run_id=plugin.run_id",
    )

    z_bias_map = straxen.URLConfig(
        infer_type=False,
        help="Map of Z bias due to non uniform drift velocity/field",
        default="legacy-z_bias://0",
    )

    def compute(self, fastsim_events):
        result = np.zeros(len(fastsim_events), dtype=self.dtype)
        # Position correction
        for alt_s2, alt, fdc in [("", "", ""), ("alt_s2_", "alt_", "_fdc")]:
            fastsim_events[f"{alt_s2}r_field_distortion_correction"] = self.fdc_map_fuse(
                np.array(
                    [
                        fastsim_events[f"{alt}s2_x"],
                        fastsim_events[f"{alt}s2_y"],
                        fastsim_events[f"{alt_s2}z_naive"],
                    ]
                ).T
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                r_cor = (
                    fastsim_events[f"{alt_s2}r_naive"]
                    + fastsim_events[f"{alt_s2}r_field_distortion_correction"]
                )
                scale = np.divide(
                    r_cor,
                    fastsim_events[f"{alt_s2}r_naive"],
                    out=np.zeros_like(r_cor),
                    where=fastsim_events[f"{alt_s2}r_naive"] != 0,
                )

            # i[f'{alt_s2}r{fdc}'] = i[f'{alt_s2}r_naive'] + i[f'{alt_s2}r_field_distortion_correction']
            # i[f'{alt_s2}x{fdc}'] = i[f'{alt_s2}r{fdc}']*np.cos(i[f'{alt_s2}theta_true'])
            # i[f'{alt_s2}y{fdc}'] = i[f'{alt_s2}r{fdc}']*np.sin(i[f'{alt_s2}theta_true'])
            result[f"{alt_s2}r{fdc}"] = r_cor
            result[f"{alt_s2}x{fdc}"] = fastsim_events[f"{alt}s2_x"] * scale
            result[f"{alt_s2}y{fdc}"] = fastsim_events[f"{alt}s2_y"] * scale
            result[f"{alt_s2}theta"] = np.arctan2(
                fastsim_events[f"{alt_s2}y{fdc}"], fastsim_events[f"{alt_s2}x{fdc}"]
            )

            # result[f'{alt_s2}z_dv_corr'] = i[f'{alt_s2}z_naive'] + resource.z_bias_map(
            #    np.array([i[f'{alt_s2}r{fdc}'], i[f'{alt_s2}z_naive']]).T,
            #    map_name='z_bias_map')
            # with np.errstate(invalid='ignore'):
            #    z_cor = -(i[f'{alt_s2}z_naive'] ** 2 - i[f'{alt_s2}r_field_distortion_correction'] ** 2) ** 0.5
            #    invalid = np.abs(i[f'{alt_s2}z_naive']) < np.abs(i[f'{alt_s2}r_field_distortion_correction'])
            #    # do not apply z correction above gate
            #    invalid |= i[f'{alt_s2}z_naive'] >= 0
            # z_cor[invalid] = i[f'{alt_s2}z_naive'][invalid]
            # i[f'{alt_s2}z_field_distortion_correction'] = z_cor - i[f'{alt_s2}z_naive']
            # i[f'{alt_s2}z'] = i[f'{alt_s2}z_naive']

        # S2 correction
        for alt_s2_interaction, alt in [("", ""), ("alt_s2_interaction_", "alt_")]:
            lifetime_corr = np.exp(
                fastsim_events[f"{alt_s2_interaction}drift_time"] / self.electron_lifetime_liquid
            )

            # S2(x,y) corrections use the observed S2 positions
            xy = np.vstack([fastsim_events[f"{alt}s2_x"], fastsim_events[f"{alt}s2_y"]]).T
            result[f"{alt}cs2"] = (
                fastsim_events[f"{alt}s2_area"] * lifetime_corr / self.s2_correction_map(xy)
            )

        return result

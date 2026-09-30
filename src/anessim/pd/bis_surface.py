"""BIS/depth response surface for propofol-opioid-sevoflurane interaction.

Nociceptive arousal is modelled as a WEAK, INCONSISTENT perturbation —
it counteracts anaesthetic depression but with small magnitude and high
noise, reflecting the different neuroanatomical pathway (spinothalamic →
thalamus → cortex arousal is partially but not fully depressed by
propofol at the thalamocortical level).

Sevoflurane sits on the SAME hypnotic axis (MAC-equivalence): 1 MAC
contributes the same u_depress as ~2.0 mcg/mL of propofol Ce.
"""

from __future__ import annotations

import numpy as np


# Population parameters from Bouillon & Bruhn interaction studies.
BIS_BASELINE = 93.0
BIS_MIN = 30.0
C50_PROP = 1.6     # mcg/mL propofol effect-site (F2 recalib: BIS real 39.6)
C50_REM = 6.5      # ng/mL remifentanil effect-site
C50_SEVO = 1.0     # MAC — 1 MAC ≈ 2.0 mcg/mL propofol for BIS
GAMMA = 1.47

# Nociceptive arousal parameters (WEAK — different pathway)
# v5.2: aún más reducidos para que el arousal (y la anti-correlación remi/prop)
# no enmascare la respuesta ce→BIS (bis_unresponsive).
BIS_AROUSAL_MAX = 4.0    # max BIS increase at full stimulus, no remi
BIS_AROUSAL_NOISE_SD = 1.0  # low noise → ce→BIS signal dominates


def bis_from_ce(
    ce_prop: float | np.ndarray,
    ce_rem: float | np.ndarray,
    sevo_mac: float | np.ndarray = 0.0,
    bis0: float = BIS_BASELINE,
    bis_min: float = BIS_MIN,
    c50_prop: float = C50_PROP,
    c50_rem: float = C50_REM,
    c50_sevo: float = C50_SEVO,
    gamma: float = GAMMA,
    nociceptive_arousal: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Predict BIS from drug effect-site concentrations and nociceptive arousal.

    Args:
        ce_prop: propofol Ce (mcg/mL).
        ce_rem: remifentanil Ce (ng/mL).
        sevo_mac: sevoflurane MAC.
        bis0: awake baseline BIS.
        bis_min: minimum achievable BIS (full depression).
        c50_prop, c50_rem, c50_sevo: C50 values (may be scaled for IIV).
        gamma: Hill coefficient.
        nociceptive_arousal: already-computed arousal drive in [0, 1].
            This should be: stimulus * remi_attenuation_bis * BIS_AROUSAL_MAX
            computed externally so the caller controls the noise injection.

    Returns:
        BIS value(s) in [0, 100].
    """
    u_prop = np.asarray(ce_prop, dtype=float) / c50_prop
    u_rem = np.asarray(ce_rem, dtype=float) / c50_rem
    u_sevo = np.asarray(sevo_mac, dtype=float) / c50_sevo

    # Depressant interaction (reduced interaction model)
    # Sevo sits on the same axis: 1 MAC ≈ u=1.0 ≈ 2.0 mcg/mL propofol
    u_depress = (
        u_prop + u_rem + u_sevo
        + 0.05 * u_prop * u_rem
        + 0.05 * u_prop * u_sevo
    )

    # Nociceptive arousal: counteracts depression
    # Small magnitude (max ~10 BIS points), high noise
    arousal = np.asarray(nociceptive_arousal, dtype=float)
    # Convert arousal [0, BIS_AROUSAL_MAX] to equivalent u units
    # At u=1.0 (C50), BIS drops from 93 to ~61.5 (half of 93-30=63).
    # So 1 BIS point ≈ 1/31.5 ≈ 0.032 u units near C50.
    u_arousal = arousal * 0.03   # ~1 BIS ≈ 0.03 u units

    u = u_depress - u_arousal
    u = np.clip(u, -5.0, None)

    effect = np.power(np.clip(u, 0, None), gamma) / (1.0 + np.power(np.clip(u, 0, None), gamma))
    bis = np.where(u >= 0, bis0 - (bis0 - bis_min) * effect, bis0)

    return float(np.clip(bis, 0, 100)) if np.isscalar(ce_prop) and np.isscalar(ce_rem) else np.clip(bis, 0, 100)


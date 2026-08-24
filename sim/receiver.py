"""Urkowitz energy detector: the `P_d` curve, and the draw that turns it into
observations.

`pd_curve` is the single most safety-critical function in the simulator, because
`agent/belief.py` **deliberately reimplements it** rather than importing this
module (DESIGN.md section 2 -- duplication is cheaper than a firewall breach) and
a cross-check test asserts the two agree to 1e-9.  It is therefore written to
match DESIGN.md section 1 literally:

    N   = dwell_s * channel_bw_hz          # complex samples
    s   = 10**(snr_eff_db / 10)            # linear SNR
    P_d = Q( (Q^-1(P_fa) - sqrt(N)*s) / (1 + s) )

The identity that removes an entire code path: **s = 0 gives
P_d = Q(Q^-1(P_fa)) = P_fa exactly.**  A silent channel therefore fires at the
false-alarm rate through the same Bernoulli draw as a live one, so there is no
separate false-alarm branch anywhere in this file -- and `P_fa` calibration and
`P_d` calibration are testing the same three lines of code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

# `scipy.stats.norm.sf(x)` is implemented as `ndtr(-x)` and `norm.isf(p)` as
# `-ndtri(p)`; both verified bit-identical here.  We call the special functions
# directly because this runs on every channel of every step and `rv_continuous`
# dispatch dominates the cost at 200k channel-dwells.
_FA_SNR_LO_DB = -24.0   # a false alarm must LOOK like a marginal weak detection,
_FA_SNR_HI_DB = -19.0   # otherwise the belief could trivially filter it out.


def pd_curve(
    snr_eff_db,
    dwell_s,
    bw_hz_per_channel: float,
    pfa: float,
):
    """Probability of detection.  Broadcasts over any of the first two arguments.

    `snr_eff_db` is the *effective* SNR: emitter SNR after the bandwidth penalty
    and any low-noise gain.  `bw_hz_per_channel` is the per-channel bandwidth
    (1 MHz), NOT the scan bandwidth -- widening the scan costs sensitivity via
    the penalty, not via the sample count.
    """
    s = np.power(10.0, np.asarray(snr_eff_db, dtype=np.float64) / 10.0)
    return pd_from_linear(s, dwell_s, bw_hz_per_channel, pfa)


def pd_from_linear(s, dwell_s, bw_hz_per_channel: float, pfa):
    """`pd_curve` in linear SNR, so a silent channel is s = 0.0 rather than -inf dB."""
    s = np.asarray(s, dtype=np.float64)
    n_samples = np.asarray(dwell_s, dtype=np.float64) * float(bw_hz_per_channel)
    thresh = -ndtri(np.asarray(pfa, dtype=np.float64))      # == norm.isf(pfa)
    return ndtr(-((thresh - np.sqrt(n_samples) * s) / (1.0 + s)))


def bw_penalty_db(bw_hz: float, db_per_octave: float) -> float:
    """Sensitivity lost by scanning wide.  DESIGN.md section 4 -- load-bearing.

    Without it the widest scan strictly dominates (same time, same energy, 20x
    the channels) and the bandwidth knob is degenerate.  At 1 dB/octave a 20 MHz
    scan is 4.32 dB less sensitive than a 1 MHz one.
    """
    return db_per_octave * float(np.log2(bw_hz / 1.0e6))


@dataclass(slots=True)
class Receiver:
    """Stateless sensing model.  All randomness is passed in, never owned."""

    pfa: float
    channel_bw_hz: float
    bw_penalty_db_per_octave: float = 1.0
    snr_est_sigma_db: float = 1.5
    gain_enabled: bool = False
    gain_db_high: float = 10.0
    gain_nf_improvement_db: float = 6.0
    gain_energy_mult: float = 1.6
    gain_saturation_snr_db: float = -5.0
    gain_fa_mult_on_saturation: float = 10.0

    @classmethod
    def from_config(cls, cfg: dict) -> "Receiver":
        rx = cfg["receiver"]
        return cls(
            pfa=float(rx["pfa"]),
            channel_bw_hz=float(cfg["grid"]["channel_bw_hz"]),
            bw_penalty_db_per_octave=float(rx.get("bw_penalty_db_per_octave", 1.0)),
            snr_est_sigma_db=float(rx.get("snr_est_sigma_db", 1.5)),
            gain_enabled=bool(rx.get("gain_enabled", False)),
            gain_db_high=float(rx.get("gain_db_high", 10.0)),
            gain_nf_improvement_db=float(rx.get("gain_nf_improvement_db", 6.0)),
            gain_energy_mult=float(rx.get("gain_energy_mult", 1.6)),
            gain_saturation_snr_db=float(rx.get("gain_saturation_snr_db", -5.0)),
            gain_fa_mult_on_saturation=float(rx.get("gain_fa_mult_on_saturation", 10.0)),
        )

    # ------------------------------------------------------------------ gain
    def gain_active(self, gain_db: float) -> bool:
        """Gain is off in every v1 config; implemented so the knob exists at all."""
        return bool(self.gain_enabled) and float(gain_db) >= self.gain_db_high

    def energy_mult(self, gain_db: float) -> float:
        return self.gain_energy_mult if self.gain_active(gain_db) else 1.0

    # --------------------------------------------------------------- sensing
    def detect_probability(
        self, rho_lin: np.ndarray, dwell_s: float, bw_hz: float, gain_db: float = 0.0
    ) -> np.ndarray:
        """Per-channel `P_d` for one scan, given time-averaged linear SNR `rho_lin`."""
        rho_lin = np.asarray(rho_lin, dtype=np.float64)
        on = self.gain_active(gain_db)

        delta_db = -bw_penalty_db(bw_hz, self.bw_penalty_db_per_octave)
        if on:
            delta_db += self.gain_nf_improvement_db
        s_eff = rho_lin * (10.0 ** (delta_db / 10.0))

        pfa = self.pfa
        if on:
            # A strong in-band signal desensitises the front end: the whole
            # in-band comb, not just the offending channel, gets a worse P_fa.
            sat = self.gain_saturation_snr_db
            if np.any(rho_lin > 10.0 ** (sat / 10.0)):
                pfa = min(self.pfa * self.gain_fa_mult_on_saturation, 1.0 - 1e-12)

        return pd_from_linear(s_eff, dwell_s, self.channel_bw_hz, pfa)

    def observe(
        self,
        rho_lin: np.ndarray,
        dwell_s: float,
        bw_hz: float,
        rng: np.random.Generator,
        gain_db: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One vectorised Bernoulli per scanned channel.

        Returns `(det_mask, reported_snr_db)`, both length `len(rho_lin)`.

        THREE fixed-size draws are taken regardless of what is actually out
        there -- the Bernoulli uniforms, the SNR-estimate noise, and the
        false-alarm SNRs.  Keeping the RNG consumption independent of truth is
        what makes the Philox-per-step-index scheme in `sim/env.py` actually
        pay: two different policies issuing the same scan at the same step index
        then see the same noise realisation.
        """
        rho_lin = np.asarray(rho_lin, dtype=np.float64)
        k = rho_lin.size
        if k == 0 or dwell_s <= 0.0:
            # A zero-length dwell collects N = 0 samples.  The Gaussian
            # approximation degenerates there (it would report P_d > P_fa off
            # zero evidence), so a truncated-to-nothing scan reports nothing.
            return np.zeros(k, dtype=bool), np.zeros(k, dtype=np.float64)

        p = self.detect_probability(rho_lin, dwell_s, bw_hz, gain_db)
        u = rng.random(k)
        est_noise = rng.standard_normal(k) * self.snr_est_sigma_db
        fa_snr = rng.uniform(_FA_SNR_LO_DB, _FA_SNR_HI_DB, size=k)

        det = u < p
        with np.errstate(divide="ignore"):
            true_db = 10.0 * np.log10(rho_lin)
        reported = np.where(rho_lin > 0.0, true_db + est_noise, fa_snr)
        return det, reported

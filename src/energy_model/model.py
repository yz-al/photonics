"""
Test 1 — Transparent energy model: joules per MAC for an optical matrix-multiply
unit versus a digital baseline, as a function of matrix dimension N.

Physical accounting (per matrix-vector product through an N x N mesh)
--------------------------------------------------------------------
One matvec = N^2 MACs.  Per matvec the optical MVM spends:

  * Conversion  ~ O(N):  N input DACs (encode the activation vector) + N output
                         ADCs (read the N detectors).  This is the term that
                         scales as N while the useful work scales as N^2, so its
                         per-MAC cost falls as 1/N.  This is the whole thesis of
                         point (a).
  * Modulation  ~ O(N):  N input modulators switch once per matvec.  (Weights are
                         set by the mesh phase shifters; in weight-stationary mode
                         they are NOT re-modulated per op.)
  * Laser       ~ O(N):  must deliver >= P_min optical power to each of N
                         detectors THROUGH the mesh, whose transmission falls
                         exponentially with mesh DEPTH (~N stages).  P_min is
                         *derived* from detector shot + thermal noise for the
                         target bit depth, not assumed.
  * Thermal     ~ O(N^2): the mesh holds ~N^2 phase shifters.  Thermo-optic
                         shifters dissipate static power continuously; per MAC
                         this is a CONSTANT floor (N^2 shifters / N^2 MACs).
                         Phase-change-material weights hold state at ~zero static
                         power -> this term collapses.  Modelled both ways.
  * Propagation ~ 0:     passive waveguide; its only energy cost is the extra
                         laser power needed to overcome loss, already in J_laser.

  J/MAC_optical(N) = [ N*(E_DAC + E_ADC + E_mod)
                       + N * E_opt_per_detector(N) / WPE
                       + N_ps(N) * P_ps * t_vec ] / N^2

Compared against a CONSTANT digital cost J_MAC_digital (independent of N), whose
matvec cost is N^2 * J_MAC_digital.

Every numeric input lives in PARAMS as a (low, nominal, high) triple with a
source string.  Uncertainty bands come from Monte-Carlo sampling those ranges.

All energies in joules, powers in watts, lengths in cm, loss in dB.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

Q_E = 1.602176634e-19  # elementary charge, C


# ----------------------------------------------------------------------------
# Parameter block.  Each entry: (low, nominal, high, "source").
# Values are reconciled against the sourced research in reports/photonic_thesis.md;
# see that file's "Inputs" table for the citation behind every triple.
# ----------------------------------------------------------------------------
@dataclass
class Param:
    low: float
    nom: float
    high: float
    source: str
    unit: str = ""

    def sample(self, rng, dist="loguniform"):
        lo, hi = self.low, self.high
        if lo <= 0 or hi <= 0 or dist == "uniform":
            return rng.uniform(lo, hi)
        return np.exp(rng.uniform(np.log(lo), np.log(hi)))


# NOTE: numbers below are reconciled with the sourced ranges (see report).
PARAMS = {
    # ---- Data converters (energy PER SAMPLE = P/f_s = FOM_W * 2^ENOB, J) -----
    # At GS/s the "good" sub-10 fJ/step envelope is unavailable; energy/sample is
    # pJ-scale.  ADC: Murmann survey + named GS/s parts (arXiv:1612.04855 4b;
    # Kull ISSCC2014 8b).  DAC: no survey exists, output-drive dominated, higher
    # and less portable than ADC (Lin ISSCC2013; CS-DAC bounds).
    "E_ADC_4b": Param(0.5e-12, 0.9e-12, 2e-12, "4b flash @GS/s; arXiv:1612.04855 (0.7pJ)"),
    "E_ADC_8b": Param(2e-12, 3.5e-12, 8e-12, "8b TI-SAR @GS/s; Kull ISSCC2014, 14nm ISSCC2018"),
    "E_DAC_4b": Param(0.5e-12, 1.5e-12, 3e-12, "4b CS-DAC @GS/s (scaled/bounded)"),
    "E_DAC_8b": Param(5e-12, 10e-12, 15e-12, "8b CS-DAC @GS/s; Lin ISSCC2013, 25GS/s part"),

    # ---- Modulators (dynamic switching energy per bit, J) --------------------
    # MZM 0.2-1 pJ typ (55-80 fJ optimized); MRM 0.9-6 fJ; plasmonic Pockels
    # 0.07-2 fJ, ITO EAM 7-20 fJ.  Sources: Xu 2014; Timurdogan Nat.Commun. 2014;
    # Heni/Hoessbacher Nat.Commun. 2019.  fJ/bit is dynamic-only (excludes laser
    # and any thermal bias); ring/plasmonic caveats handled in laser+thermal terms.
    "E_mod_MZM":  Param(55e-15, 400e-15, 1000e-15, "Si MZM dynamic; Xu2014, opt. slow-light"),
    "E_mod_MRM":  Param(0.9e-15, 3e-15, 6e-15, "Si micro-ring; Timurdogan Nat.Commun.2014"),
    "E_mod_plasmonic": Param(0.07e-15, 1e-15, 20e-15, "plasmonic Pockels..ITO EAM; Heni2019"),

    # ---- Thermal phase shifter STATIC power (W per shifter, held avg 0.5*Ppi) -
    # Ppi 0.6-30 mW/pi; typical SOI ~10-25 mW/pi (Opt.Lett.45,4806 2020).
    # A shifter parked at random phase holds ~0.5*Ppi continuously.  Sub-mW/pi
    # needs suspension -> kHz bandwidth (unusable for GS/s), so excluded from low.
    "P_ps_thermal": Param(3e-3, 10e-3, 25e-3, "thermo-optic Ppi; OptLett 45,4806 (2020)"),
    "P_ps_pcm":     Param(0.0, 1e-10, 1e-9, "PCM (Sb2Se3) non-volatile: ~0 W hold; Delaney SciAdv2021"),

    # ---- Laser / detector link ----------------------------------------------
    "WPE_laser": Param(0.05, 0.15, 0.40, "integrated DFB laser wall-plug efficiency"),
    "responsivity": Param(0.8, 1.0, 1.1, "Ge-on-Si photodiode A/W @1550nm"),
    "tia_noise": Param(2e-12, 5e-12, 20e-12, "TIA input-referred current noise A/rtHz"),
    "bandwidth": Param(1e9, 10e9, 50e9, "per-channel symbol bandwidth Hz"),

    # ---- Optical loss --------------------------------------------------------
    "wg_loss_dbcm": Param(0.5, 1.5, 3.0, "Si strip prop. loss; Dumon OptExpress2004 (3.6), typ 1-2"),
    "coupling_db":  Param(0.5, 1.5, 3.0, "per-facet coupling; edge<1, grating 1-3 dB"),
    "mzi_loss_db":  Param(0.1, 0.5, 2.0, "per-MZI IL; Bandyopadhyay Optica2021 (0.5), realized up to 2"),

    # ---- Geometry ------------------------------------------------------------
    "mzi_cell_len_um": Param(100.0, 250.0, 500.0, "MZI cell length along propagation (um)"),
    "mzi_port_pitch_um": Param(25.0, 80.0, 110.0, "port pitch, thermal-crosstalk limited (um)"),

    # ---- Digital baseline (J per MAC), whole-chip REALIZED --------------------
    # 1 MAC = 2 ops; J/MAC = 2/(TOPS/W*1e12).  H100 INT8 0.71pJ, B200 INT8 0.44pJ,
    # B200 FP4 0.20pJ (peak); realized ~1.5-3x worse (Horowitz ISSCC2014 floor
    # ~0.23pJ@45nm -> ~20-40fJ@5nm arithmetic-only; CIM macro 10-50fJ).
    "J_MAC_digital_8b": Param(0.3e-12, 0.5e-12, 2.0e-12, "INT8 H100/B200 whole-chip realized"),
    "J_MAC_digital_4b": Param(0.2e-12, 0.3e-12, 1.0e-12, "FP4 B200 whole-chip realized"),
    # Hard target: the arithmetic/CIM floor optics must beat to be a compute win.
    "J_MAC_digital_floor": Param(20e-15, 40e-15, 100e-15, "5nm INT8 arithmetic / CIM macro floor"),

    # ---- Microring crossbar (weight-bank / broadcast-and-weight) --------------
    # RECONCILED with sourced ranges; see report "Crossbar inputs" table.
    # Per-ring continuous thermal-lock power; athermal designs reduce the low end.
    # active tuning ~2-30 mW/ring (undercut ~2-5); low end 0.5 = athermal-optimistic
    # (flagged: passive athermal conflicts with a *tunable* weight ring). Padmaraju&
    # Bergman 2014; Tait 2016/2020; Guha OptExpress 2010/2013.
    # 1 mW avg (Feldmann2021/Nahmias2020) .. 20-46 mW full-FSR trim (undercut lit.)
    "ring_stab_mW":    Param(1e-3, 6e-3, 30e-3, "per-ring thermal hold; Feldmann/Nahmias..full-FSR"),
    "ring_through_db": Param(0.01, 0.05, 0.2, "ring off-res through-port loss on bus"),
    "ring_Q":          Param(5e3, 1e4, 2e4, "weight-bank ring loaded Q; Tait2016 (~1e4)"),
    "ring_radius_um":  Param(5.0, 10.0, 20.0, "ring radius -> FSR (18/9/4.6 nm)"),
    "fab_scatter_nm":  Param(0.1, 0.5, 2.0, "resonance scatter; Selvaraja2010, Lu2017 (~1nm wafer)"),
    "tune_nm_per_mW":  Param(0.05, 0.15, 0.3, "heater tuning efficiency"),
    # channel spacing = 3.4-4.6 linewidths for 1-pole banks at 3 dB penalty
    "chan_spacing_factor": Param(3.4, 4.0, 4.6, "spacing/linewidth; Tait IPC 2017 (8116022)"),
    # trim range >= 1 FSR mandatory (Selvaraja/Milanizadeh); heaters provide several nm
    "ring_trim_range_nm":  Param(3.0, 6.0, 12.0, "heater trim range >=1 FSR; Milanizadeh JLT2021"),
    "digital_add_J":   Param(2e-15, 5e-15, 15e-15, "digital accumulate per partial sum @5nm"),
    "n_group_si":      Param(4.0, 4.2, 4.4, "Si waveguide group index @1550"),
    # achievable ring weight precision (Lorentzian slope + drift limited): ~4-5 bits
    # -> 8-bit weights are NOT physically available from a weight ring (failure mode 5)
    "ring_weight_bits": Param(3.1, 4.0, 5.1, "ring weight precision; Tait2016/2018, Feldmann2021"),
}


def _pget(name):
    return PARAMS[name]


# ----------------------------------------------------------------------------
# Derived optical quantities
# ----------------------------------------------------------------------------
def required_detector_energy(bits, responsivity, tia_noise, bandwidth):
    """
    Minimum OPTICAL energy per detected OUTPUT symbol (J) for a target bit depth,
    DERIVED from detector noise rather than assumed.

    Requirement (standard full-scale converter DR): SNR_dB = 6.02*bits + 1.76,
    so linear POWER-SNR = 10**(SNR_dB/10).  (Anderson et al., "Optical Transformers"
    arXiv:2302.10360, use SNR=sqrt(N_photons); ADI MT-001 for the DR formula.)

    Two noise floors; the larger required power wins:
      * shot noise: sigma_i^2 = 2 q I B, I = R P
            SNR = I^2/(2 q I B) = R P/(2 q B)
            -> E = P/B = SNR * 2 q / R           (B-independent)   [matches ~0.05fJ@4b]
      * thermal/TIA noise: sigma_i^2 = i_n^2 B
            SNR = (R P)^2/(i_n^2 B)
            -> E = P/B = sqrt(SNR) * i_n / (R sqrt(B))              [matches ~3fJ@4b, 49fJ@8b]

    Validated against sourced link-budget: 3.1 fJ @4b and 49 fJ @8b at 1 GHz,
    TIA-limited (i_n=5 pA/rtHz).  At these powers the receiver is thermal-limited.
    """
    snr = 10.0 ** ((6.02 * bits + 1.76) / 10.0)   # linear power SNR
    E_shot = snr * 2 * Q_E / responsivity
    P_th = np.sqrt(snr) * tia_noise * np.sqrt(bandwidth) / responsivity
    E_th = P_th / bandwidth
    return np.maximum(E_shot, E_th)


def mesh_depth(N, arch, tile):
    """Optical depth (number of MZI stages light traverses in series).

    monolithic: a single Clements universal N x N mesh has depth N.  A general
        (non-unitary) real matrix needs U.Sigma.V (two meshes) ~ depth 2N; we use
        N as the OPTIMISTIC (unitary) case -- favourable to optics.
    tiled: the matrix is blocked into T x T sub-meshes; light only ever traverses
        one tile's depth T, then is detected/re-converted at the tile boundary.
        Depth is bounded at T regardless of N -> loss is bounded, but the O(N)
        conversion amortisation is capped at T (see optical_j_per_mac).
    butterfly: an FFT-style mesh has log2(N) stages of N/2 2x2 MZIs.  Optical
        depth = log2(N) -> loss stops being the wall.  BUT the transform does only
        ~N*log2(N) MACs (not N^2), while still needing N input + N output
        conversions, so conversion amortises over only log2(N), not N."""
    if arch == "tiled":
        return min(N, tile)
    if arch == "butterfly":
        return max(1, int(np.ceil(np.log2(N))))
    return N


def amortisation_dim(N, arch, tile):
    """MACs-per-conversion amortisation dimension A.

    per-MAC conversion cost = (2 conversions per I/O channel) / A, where
      dense/monolithic:  N^2 MACs / (2N conversions)  -> A = N
      tiled:             re-converts every T-block      -> A = min(N, T)
      butterfly:         N*log2(N) MACs / (2N conv.)    -> A = log2(N)
    The butterfly's A = log2(N) is the crux: it grows only logarithmically."""
    if arch == "tiled":
        return max(1.0, float(min(N, tile)))
    if arch == "butterfly":
        return max(1.0, float(np.log2(N)))
    return float(N)


def mesh_transmission_db(depth, coupling_db, mzi_loss_db, wg_loss_dbcm, cell_len_um):
    """Total insertion loss (dB) through a mesh of given optical depth:
    2 facet couplers + depth*per-MZI IL + propagation over depth*cell_len."""
    prop_len_cm = depth * cell_len_um * 1e-4  # um -> cm
    return 2 * coupling_db + depth * mzi_loss_db + prop_len_cm * wg_loss_dbcm


def n_phase_shifters(N, arch="monolithic"):
    """Phase-shifter count.
    Clements/monolithic: N(N-1)/2 MZIs, ~2 shifters each -> ~N(N-1).
    butterfly: log2(N) stages x N/2 MZIs -> (N/2)log2(N) MZIs, ~N*log2(N) shifters."""
    if arch == "butterfly":
        return N * max(1, int(np.ceil(np.log2(N))))
    return N * (N - 1)


def n_mzi(N, arch="monolithic"):
    if arch == "butterfly":
        return (N // 2) * max(1, int(np.ceil(np.log2(N))))
    return N * (N - 1) // 2


# Usable analog optical link budget (dB): ~0 dBm launch to ~ -30..-40 dBm
# sensitivity at GHz rates.  Beyond this the mesh output is undetectable.
LINK_BUDGET_DB = 33.0


# ----------------------------------------------------------------------------
# Core per-MAC energy model
# ----------------------------------------------------------------------------
def optical_j_per_mac(N, bits, p, weight_stationary=True, modulator="MRM",
                      arch="monolithic", tile=64, enforce_loss=False, ignore_loss=False):
    """J/MAC for the optical MVM at matrix size N and given bit depth.

    arch: 'monolithic' (one depth-N mesh) or 'tiled' (T x T sub-meshes).
        In the tiled case conversion amortises over the tile dimension T, not N:
        each T-wide tile row needs its own T input DACs + T output ADCs, so the
        per-MAC conversion floor is set by T.  This is the physically buildable
        architecture and it is why the O(N) amortisation does NOT run to large N.
    enforce_loss: if True, return np.inf when mesh loss exceeds LINK_BUDGET_DB
        (signal undetectable) -- makes the loss wall explicit in sweeps.
    weight_stationary: weights in PCM (~0 static, no per-op weight DAC) vs
        thermo-optic (continuous ~0.5*Ppi hold + per-op weight reprogram).
    """
    E_ADC = p[f"E_ADC_{bits}b"]
    E_DAC = p[f"E_DAC_{bits}b"]
    E_mod = {"MZM": p["E_mod_MZM"], "MRM": p["E_mod_MRM"],
             "plasmonic": p["E_mod_plasmonic"]}[modulator]

    amort = amortisation_dim(N, arch, tile)  # MACs-per-conversion amortisation dim

    # --- Conversion + modulation: per-MAC = (E_DAC+E_ADC+E_mod)/amort.
    #     dense A=N ; tiled A=T ; butterfly A=log2(N)  (N*log2N MACs / 2N conversions)
    E_conv_mod_perMAC = (E_DAC + E_ADC + E_mod) / amort

    # --- Laser: deliver E_det to each detector through the mesh depth.
    E_det = required_detector_energy(bits, p["responsivity"], p["tia_noise"], p["bandwidth"])
    depth = mesh_depth(N, arch, tile)
    if ignore_loss:
        trans = 1.0
    else:
        loss_db = mesh_transmission_db(depth, p["coupling_db"], p["mzi_loss_db"],
                                       p["wg_loss_dbcm"], p["mzi_cell_len_um"])
        if enforce_loss and loss_db > LINK_BUDGET_DB:
            return np.inf
        # clamp to avoid float underflow; >~300 dB is already undetectable
        trans = 10.0 ** (-min(loss_db, 300.0) / 10.0)
    # per-MAC laser: each detected OUTPUT symbol (there are amort of them per tile,
    # doing amort^2 MACs per tile) costs E_det/trans/WPE -> per-MAC = E_det/(amort*trans*WPE)
    E_laser_perMAC = E_det / (amort * trans * p["WPE_laser"])

    # --- Thermal: static shifter power amortised over throughput.
    #     Per MAC = (P_ps per shifter) * t_vec, since #shifters ~ #MACs per matvec.
    t_vec = 1.0 / p["bandwidth"]
    if weight_stationary:
        P_ps = p["P_ps_pcm"]
        weight_reprog_perMAC = 0.0
    else:
        P_ps = p["P_ps_thermal"] * 0.5  # avg hold at random phase ~ 0.5*Ppi
        weight_reprog_perMAC = E_DAC   # reprogram each weight every matvec (O(N^2)/N^2)
    E_thermal_perMAC = P_ps * t_vec

    return E_conv_mod_perMAC + E_laser_perMAC + E_thermal_perMAC + weight_reprog_perMAC


def optical_terms(N, bits, p, weight_stationary=True, modulator="MRM",
                  arch="monolithic", tile=64):
    """Return the individual per-MAC terms (for the dominance analysis)."""
    E_ADC = p[f"E_ADC_{bits}b"]; E_DAC = p[f"E_DAC_{bits}b"]
    E_mod = {"MZM": p["E_mod_MZM"], "MRM": p["E_mod_MRM"],
             "plasmonic": p["E_mod_plasmonic"]}[modulator]
    amort = amortisation_dim(N, arch, tile)
    E_det = required_detector_energy(bits, p["responsivity"], p["tia_noise"], p["bandwidth"])
    depth = mesh_depth(N, arch, tile)
    loss_db = mesh_transmission_db(depth, p["coupling_db"], p["mzi_loss_db"],
                                   p["wg_loss_dbcm"], p["mzi_cell_len_um"])
    trans = 10.0 ** (-loss_db / 10.0)
    t_vec = 1.0 / p["bandwidth"]
    if weight_stationary:
        P_ps = p["P_ps_pcm"]; wr = 0.0
    else:
        P_ps = p["P_ps_thermal"] * 0.5; wr = E_DAC
    return {
        "conversion": (E_DAC + E_ADC) / amort + wr,
        "modulation": E_mod / amort,
        "laser": E_det / (amort * trans * p["WPE_laser"]),
        "thermal": P_ps * t_vec,
        "loss_db": loss_db,
    }


def digital_j_per_mac(bits, p):
    return p[f"J_MAC_digital_{bits}b"]


LAMBDA_NM = 1550.0


def crossbar_K_max(p):
    """Achievable WDM channel count K = FSR / channel_spacing, derived.

    FSR = lambda^2/(n_g * 2*pi*R); channel_spacing = factor * linewidth,
    linewidth = lambda/Q.  K is the number of independent inputs carried in
    parallel -> it caps the conversion amortisation at K, not N."""
    R_nm = p["ring_radius_um"] * 1e3
    fsr_nm = LAMBDA_NM**2 / (p["n_group_si"] * 2 * np.pi * R_nm)
    linewidth_nm = LAMBDA_NM / p["ring_Q"]
    spacing_nm = p["chan_spacing_factor"] * linewidth_nm
    return max(1.0, np.floor(fsr_nm / spacing_nm))


def crossbar_yield(N, p):
    """Array yield = (per-ring yield)^(N^2), per-ring = P(|fab scatter| < trim range).
    Fab scatter ~ Gaussian(sigma = fab_scatter_nm)."""
    from math import erf
    sigma = p["fab_scatter_nm"]
    tr = p["ring_trim_range_nm"]
    per_ring = erf(tr / (np.sqrt(2) * sigma)) if sigma > 0 else 1.0
    n_rings = N * N
    # log to avoid underflow
    log_yield = n_rings * np.log(max(per_ring, 1e-300))
    return {"per_ring": per_ring, "array_log10_yield": log_yield / np.log(10),
            "array_yield": np.exp(log_yield)}


def crossbar_j_per_mac(N, bits, p, weight_stationary=True, enforce_loss=False,
                       ignore_loss=False):
    """J/MAC for a microring crossbar weight bank (broadcast-and-weight).

    N x N array, N^2 rings (weight-stationary).  K = min(N, K_max) wavelengths
    carry K inputs in parallel; contraction is tiled in ceil(N/K) time passes
    with DIGITAL accumulation of partial sums (charged).

    Per N x N matvec (N^2 MACs):
      DAC  : N (load each input once)          -> E_DAC/N per MAC
      mod  : N ring modulations                -> E_mod/N per MAC
      ADC  : N^2/K detections                  -> E_ADC/K per MAC
      dadd : N^2/K digital accumulations       -> E_add/K per MAC
      laser: E_det per detection through K-ring bus, /K amortised
      therm: N^2 rings * P_stab, throughput N*K*bw -> N*P_stab/(K*bw) per MAC
    """
    E_ADC = p[f"E_ADC_{bits}b"]; E_DAC = p[f"E_DAC_{bits}b"]
    E_mod = p["E_mod_MRM"]  # ring modulator
    K = min(float(N), crossbar_K_max(p))

    conv = E_DAC / N + E_mod / N + E_ADC / K + p["digital_add_J"] / K

    # bus loss: light passes K rings (through-port) + 2 couplers
    bus_loss_db = 2 * p["coupling_db"] + K * p["ring_through_db"]
    if ignore_loss:
        trans = 1.0
    else:
        if enforce_loss and bus_loss_db > LINK_BUDGET_DB:
            return np.inf
        trans = 10.0 ** (-min(bus_loss_db, 300.0) / 10.0)
    E_det = required_detector_energy(bits, p["responsivity"], p["tia_noise"], p["bandwidth"])
    laser = E_det / (K * trans * p["WPE_laser"])

    # thermal stabilisation: N^2 rings continuously locked (PCM holds the WEIGHT
    # but the RESONANCE still drifts -> lock power remains; athermal cuts the low end)
    P_stab = p["ring_stab_mW"]
    thermal = N * P_stab / (K * p["bandwidth"])
    # weight reprogram: PCM stationary -> 0; thermally-tuned weights -> reload per pass
    weight_reprog = 0.0 if weight_stationary else E_DAC / K

    return conv + laser + thermal + weight_reprog


def crossbar_terms(N, bits, p, weight_stationary=True):
    E_ADC = p[f"E_ADC_{bits}b"]; E_DAC = p[f"E_DAC_{bits}b"]; E_mod = p["E_mod_MRM"]
    K = min(float(N), crossbar_K_max(p))
    bus_loss_db = 2 * p["coupling_db"] + K * p["ring_through_db"]
    trans = 10.0 ** (-min(bus_loss_db, 300.0) / 10.0)
    E_det = required_detector_energy(bits, p["responsivity"], p["tia_noise"], p["bandwidth"])
    return {
        "K": K,
        "conversion": E_DAC / N + E_mod / N + E_ADC / K + p["digital_add_J"] / K,
        "laser": E_det / (K * trans * p["WPE_laser"]),
        "thermal_stab": N * p["ring_stab_mW"] / (K * p["bandwidth"]),
        "bus_loss_db": bus_loss_db,
    }


def crossbar_feasibility(N, p=None):
    if p is None:
        p = {k: v.nom for k, v in PARAMS.items()}
    K = min(float(N), crossbar_K_max(p))
    n_rings = N * N
    bus_loss = {
        "best_0p005": 2 * p["coupling_db"] + K * 0.005,
        "typ_0p02":   2 * p["coupling_db"] + K * 0.02,
        "high_0p05":  2 * p["coupling_db"] + K * 0.05,
    }
    y = crossbar_yield(N, p)
    # area: N^2 rings at a ring pitch; ring pitch ~ 2*radius + gap ~ port pitch
    ring_pitch_um = max(p["mzi_port_pitch_um"], 2 * p["ring_radius_um"] + 5)
    area_cm2 = (n_rings) * (ring_pitch_um * 1e-4) ** 2
    idle_stab_W = n_rings * p["ring_stab_mW"]
    return {
        "N": N, "arch": "crossbar", "K_max": crossbar_K_max(p), "K_used": K,
        "n_rings": n_rings, "bus_rings": int(K), "bus_loss_db": bus_loss,
        "detectable_typ": bus_loss["typ_0p02"] <= LINK_BUDGET_DB,
        "per_ring_yield": y["per_ring"], "array_log10_yield": y["array_log10_yield"],
        "area_cm2": area_cm2, "idle_stab_kW": idle_stab_W / 1e3,
    }


def feasibility(N, p=None, arch="monolithic", tile=64):
    """Physical feasibility of realising the linear op at size N.

    Reports MZI/phase-shifter counts, die area, worst/typ/best mesh insertion
    loss through the optical depth, and the thermo-optic idle tuning power.
    For 'tiled', reports per-tile depth-loss (the buildable quantity) and the
    number of tiles needed to cover an N x N matrix."""
    if p is None:
        p = {k: v.nom for k, v in PARAMS.items()}
    nmzi = n_mzi(N, arch)
    n_ps = n_phase_shifters(N, arch)
    depth = mesh_depth(N, arch, tile)
    # area: width = N * port_pitch ; length = depth * cell_len
    width_cm = N * p["mzi_port_pitch_um"] * 1e-4
    length_cm = depth * p["mzi_cell_len_um"] * 1e-4
    area_cm2 = width_cm * length_cm
    loss = {
        "best_0p1": mesh_transmission_db(depth, p["coupling_db"], 0.1, p["wg_loss_dbcm"], p["mzi_cell_len_um"]),
        "good_0p5": mesh_transmission_db(depth, p["coupling_db"], 0.5, p["wg_loss_dbcm"], p["mzi_cell_len_um"]),
        "typ_2p0":  mesh_transmission_db(depth, p["coupling_db"], 2.0, p["wg_loss_dbcm"], p["mzi_cell_len_um"]),
    }
    idle_tuning_W = n_ps * p["P_ps_thermal"] * 0.5  # thermo-optic, avg 0.5*Ppi
    RETICLE_MM2 = 858.0
    WAFER300_MM2 = 70686.0
    n_tiles = (max(1, N // tile)) ** 2 if arch == "tiled" else 1
    return {
        "N": N, "arch": arch, "tile": tile if arch == "tiled" else None,
        "n_mzi": nmzi, "n_phase_shifters": n_ps, "optical_depth": depth,
        "area_cm2": area_cm2, "area_mm2": area_cm2 * 100,
        "reticles": area_cm2 * 100 / RETICLE_MM2,
        "wafers_300mm": area_cm2 * 100 / WAFER300_MM2,
        "loss_db": loss, "detectable_best": loss["best_0p1"] <= LINK_BUDGET_DB,
        "idle_tuning_W": idle_tuning_W, "n_tiles": n_tiles,
    }


# ----------------------------------------------------------------------------
# Monte-Carlo over parameter ranges -> uncertainty bands
# ----------------------------------------------------------------------------
def sample_params(rng):
    return {k: v.sample(rng) for k, v in PARAMS.items()}


def sweep(Ns, bits, n_mc=2000, weight_stationary=True, modulator="MRM", seed=0,
          arch="monolithic", tile=64, enforce_loss=False, digital_key=None,
          ignore_loss=False):
    """Return dict with optical/digital J/MAC percentile bands vs N."""
    rng = np.random.default_rng(seed)
    opt = np.empty((n_mc, len(Ns)))
    dig = np.empty((n_mc, len(Ns)))
    crossings = []
    dkey = digital_key or f"J_MAC_digital_{bits}b"
    for i in range(n_mc):
        p = sample_params(rng)
        o = np.array([optical_j_per_mac(N, bits, p, weight_stationary, modulator,
                                        arch, tile, enforce_loss, ignore_loss) for N in Ns])
        d = np.full(len(Ns), p[dkey])
        opt[i] = o
        dig[i] = d
        # crossover: smallest N where optical <= digital
        below = np.where(o <= d)[0]
        crossings.append(Ns[below[0]] if len(below) else np.inf)
    return {
        "Ns": np.array(Ns),
        "opt_p50": np.percentile(opt, 50, axis=0),
        "opt_p16": np.percentile(opt, 16, axis=0),
        "opt_p84": np.percentile(opt, 84, axis=0),
        "opt_p05": np.percentile(opt, 5, axis=0),
        "opt_p95": np.percentile(opt, 95, axis=0),
        "dig_p50": np.percentile(dig, 50, axis=0),
        "dig_p16": np.percentile(dig, 16, axis=0),
        "dig_p84": np.percentile(dig, 84, axis=0),
        "crossings": np.array(crossings, dtype=float),
        "opt_all": opt,
        "dig_all": dig,
    }


def crossover_band(res):
    c = res["crossings"]
    finite = c[np.isfinite(c)]
    frac_exists = len(finite) / len(c)
    if len(finite) == 0:
        return {"exists_frac": 0.0, "p16": np.inf, "p50": np.inf, "p84": np.inf}
    return {
        "exists_frac": frac_exists,
        "p05": np.percentile(finite, 5),
        "p16": np.percentile(finite, 16),
        "p50": np.percentile(finite, 50),
        "p84": np.percentile(finite, 84),
        "p95": np.percentile(finite, 95),
    }

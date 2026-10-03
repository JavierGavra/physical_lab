#!/usr/bin/env python3
"""BPSK decision-sample simulation plus an illustrative link budget."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bit_error(gamma):
    return 0.5 * math.erfc(math.sqrt(gamma))


def simulate(a, out):
    rng = np.random.default_rng(a.seed)
    bits = rng.integers(0, 2, a.bits)
    symbols = 2.0 * bits - 1.0
    rows = []
    for db in range(-2, 11, 2):
        gamma = 10 ** (db / 10)
        sigma = math.sqrt(1 / (2 * gamma))
        noisy = symbols + rng.normal(0, sigma, a.bits)
        errors = int(np.count_nonzero((noisy >= 0) != bits))
        # Three transmissions, same TOTAL energy per information bit.
        repeated = symbols[:, None] / math.sqrt(3)
        received = repeated + rng.normal(0, sigma, (a.bits, 3))
        decoded = np.sum(received >= 0, axis=1) >= 2
        rep_errors = int(np.count_nonzero(decoded != bits))
        p = bit_error(gamma / 3)
        rep_theory = 3 * p * p * (1 - p) + p ** 3
        rows.append([db, errors, errors / a.bits, bit_error(gamma),
                     rep_errors, rep_errors / a.bits, rep_theory])
    with open(out / "ber.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ebn0_db", "errors", "ber", "theory",
                    "rep3_errors", "rep3_ber", "rep3_theory"])
        w.writerows(rows)
    data = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogy(data[:, 0], data[:, 3], "-", label="Uncoded theory")
    nonzero = data[:, 2] > 0
    ax.semilogy(data[nonzero, 0], data[nonzero, 2], "o",
                label="Uncoded simulation")
    zeros = ~nonzero
    if np.any(zeros):
        ax.semilogy(data[zeros, 0], np.full(np.sum(zeros), 3 / a.bits),
                    "v", label="Zero errors: approximate 95% upper bound")
    ax.semilogy(data[:, 0], data[:, 6], "--", label="Repetition-3 theory")
    ax.semilogy(data[:, 0], data[:, 5], "s", label="Repetition-3 simulation")
    ax.set(xlabel="Eb/N0 per information bit (dB)",
           ylabel="Bit error probability / observed BER",
           title="BPSK in AWGN; equal energy per information bit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "ber.png", dpi=180)
    plt.close(fig)
    # Symbol-decision samples; this is not a sampled RF carrier waveform.
    sample_bits = bits[:32]
    sample_tx = 2.0 * sample_bits - 1.0
    sample_rx = sample_tx + rng.normal(0, math.sqrt(1 / 2), 32)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.step(np.arange(32), sample_tx, where="mid", label="Ideal symbols")
    ax.plot(np.arange(32), sample_rx, "o", label="Noisy decisions, Eb/N0 = 0 dB")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Symbol index", ylabel="Normalized decision amplitude",
           title="BPSK decision samples and zero threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "symbols.png", dpi=180)
    plt.close(fig)
    for row in rows:
        print(f"Eb/N0={row[0]:2d} dB; errors={row[1]:6d}; BER={row[2]:.6g}")


def budget(a, out):
    # Illustrative assumptions: d0 = 1 m; PL(d0) = 40 dB; gains = 0 dBi.
    # 290 K thermal noise density, 125 kHz noise bandwidth, NF = 6 dB.
    bandwidth = 125_000
    noise_dbm = -174 + 10 * math.log10(bandwidth) + 6
    rows = []
    for d in np.geomspace(1, 3000, 100):
        loss = 40 + 10 * a.path_exponent * math.log10(d)
        rx = a.tx_dbm - loss
        snr = rx - noise_dbm
        ebn0 = snr + 10 * math.log10(bandwidth / a.rate)
        ber = bit_error(10 ** (ebn0 / 10))
        packet_error = -math.expm1(256 * math.log1p(-ber))
        rows.append([d, loss, rx, snr, ebn0, ber, packet_error])
    with open(out / "link_budget.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_m", "path_loss_db", "rx_dbm", "snr_db",
                    "ebn0_db", "ber", "packet_error_256bits"])
        w.writerows(rows)
    data = np.asarray(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].semilogx(data[:, 0], data[:, 4])
    axes[0].set(xlabel="Distance (m)", ylabel="Eb/N0 (dB)")
    axes[1].semilogx(data[:, 0], data[:, 6])
    axes[1].set(xlabel="Distance (m)", ylabel="256-bit packet error probability",
                ylim=(-0.03, 1.03))
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Illustrative log-distance model; not measured radio range")
    fig.tight_layout()
    fig.savefig(out / "link_budget.png", dpi=180)
    plt.close(fig)
    print("Assumed receiver noise (dBm):", round(noise_dbm, 2))
    print("256-bit payload airtime (ms):", 1000 * 256 / a.rate)
    print("TX circuit energy at 50 mW (mJ):", 50 * 256 / a.rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tx-dbm", type=float, default=0)
    p.add_argument("--path-exponent", type=float, default=2.7)
    p.add_argument("--rate", type=float, default=10_000)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    if a.bits < 32 or a.rate <= 0 or a.path_exponent <= 0:
        p.error("bits >= 32; rate and path exponent must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    simulate(a, out)
    budget(a, out)
    print("Outputs:", out.resolve())


if __name__ == "__main__":
    main()


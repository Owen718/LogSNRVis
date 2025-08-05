#!/usr/bin/env python3
"""
Plot log‑SNR distributions as in Fig. 2 of:
  "Style‑Friendly SNR Sampler for Style‑Driven Generation" (arXiv:2411.14793).

It reproduces four Normal PDFs in λ (logSNR) space:
  1) SD3/FLUX training sampler in λ-space:  N(μ = -2*log(3), σ = 2)
  2) Style‑friendly: N(μ = -6, σ = 1)
  3) Style‑friendly: N(μ = -6, σ = 2)
  4) Style‑friendly: N(μ = -6, σ = 3)

You can customize curves via --curves JSON (list of {mu, sigma, label}).
The shaded region defaults to λ <= -5.

Usage:
  python plot_style_friendly_logsnr.py --out example.png
  python plot_style_friendly_logsnr.py --xmin -12.5 --xmax 7.5 --shade -5 \
      --curves '[{"mu":-2*math.log(3),"sigma":2,"label":"SD3 and Flux"}, {"mu":-6,"sigma":1,"label":"μ=-6, σ=1"}]'
"""
import argparse, math, json
import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma):
    x = np.asarray(x)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xmin", type=float, default=-12.5)
    p.add_argument("--xmax", type=float, default=7.5)
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--shade", type=float, default=-5.0, help="shade λ <= this value")
    p.add_argument("--title", type=str, default="Probability Distribution of logSNR")
    p.add_argument("--xlabel", type=str, default="logSNR")
    p.add_argument("--ylabel", type=str, default="Probability Density")
    p.add_argument("--out", type=str, default="example.png")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--curves", type=str, default="")
    args = p.parse_args()

    # Default curves (replicate paper)
    curves = [
        {"mu": -2.0*math.log(3.0), "sigma": 2.0, "label": "SD3 and Flux"},
        {"mu": -6.0, "sigma": 1.0, "label": r"$\mu=-6$ and $\sigma=1$"},
        {"mu": -6.0, "sigma": 2.0, "label": r"Style friendly ($\sigma=2$)"},
        {"mu": -6.0, "sigma": 3.0, "label": r"Style friendly ($\sigma=3$)"},
    ]
    if args.curves:
        # Allow simple Python expressions inside numbers (e.g., -2*math.log(3))
        data = json.loads(args.curves)
        out = []
        for c in data:
            mu = float(eval(str(c["mu"]), {"__builtins__":{},"math":math}))
            sigma = float(eval(str(c["sigma"]), {"__builtins__":{},"math":math}))
            label = str(c.get("label", f"mu={mu}, sigma={sigma}"))
            out.append({"mu":mu,"sigma":sigma,"label":label})
        curves = out

    x = np.linspace(args.xmin, args.xmax, args.num)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Shaded high-noise region
    if args.shade is not None:
        ax.axvspan(args.xmin, args.shade, alpha=0.2)

    for c in curves:
        y = normal_pdf(x, c["mu"], c["sigma"])
        ax.plot(x, y, label=c["label"])

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()

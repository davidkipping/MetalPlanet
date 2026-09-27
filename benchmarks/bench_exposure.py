"""Exposure averaging: contact-split Gauss-Legendre vs supersampling.

Uniform supersampling converges as O(1/N) because the light curve's
derivative jumps at each contact. Splitting the exposure window at the
contacts and applying a fixed-order Gauss-Legendre rule to each smooth
piece converges geometrically. Writes exposure.json.
"""
import json
import os

import numpy as np

import metalplanet

HERE = os.path.dirname(os.path.abspath(__file__))
T = np.linspace(-0.13, 0.13, 261)
EXP = 0.02          # ~29 min; T14 here is 0.126 d, so smearing is heavy


def model(**kw):
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.456, 0.1, 8.8, 87.07
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4, 0.25], "quadratic"
    return metalplanet.TransitModel(p, T, **kw), p


def main():
    m1, p1 = model(exp_time=EXP, supersample_factor=200001)
    m2, p2 = model(exp_time=EXP, integration="contact", n_gl=40)
    a, b = m1.light_curve(p1), m2.light_curve(p2)
    agree = float(np.abs(a - b).max())
    print(f"reference cross-check (200,001-point supersample vs 40-point "
          f"contact GL): {agree:.3e}\n")
    ref = b

    rows = []
    print(f"{'method':<26}{'evals/exposure':>16}{'max |err|':>14}")
    for n in (3, 7, 11, 31, 101, 1001, 10001):
        m, p = model(exp_time=EXP, supersample_factor=n)
        err = float(np.abs(m.light_curve(p) - ref).max())
        rows.append({"method": "supersample", "n": n, "evals": n,
                     "max_err": err})
        print(f"{'supersample N=' + str(n):<26}{n:>16}{err:>14.3e}")
    for n in (2, 3, 5, 7, 11):
        m, p = model(exp_time=EXP, integration="contact", n_gl=n)
        err = float(np.abs(m.light_curve(p) - ref).max())
        rows.append({"method": "contact_gl", "n": n, "evals": 5 * n,
                     "max_err": err})
        print(f"{'contact GL n=' + str(n):<26}{5 * n:>16}{err:>14.3e}")

    gl5 = next(r for r in rows if r["method"] == "contact_gl" and r["n"] == 5)
    ss = [r for r in rows if r["method"] == "supersample"]
    match = [r for r in ss if r["max_err"] <= gl5["max_err"]]
    if match:
        cheapest = min(match, key=lambda r: r["evals"])
        need = cheapest["evals"]
        how = "measured"
    else:
        # supersampling is O(1/N) here, so extrapolate from the largest N
        big = max(ss, key=lambda r: r["evals"])
        need = big["evals"] * big["max_err"] / gl5["max_err"]
        how = f"extrapolated from N={big['evals']:,} (error ~ 1/N)"
    print(f"\ncontact GL n=5 reaches {gl5['max_err']:.2e} with "
          f"{gl5['evals']} evaluations per exposure; uniform supersampling "
          f"needs N ~ {need:,.0f} for the same accuracy ({how}) — "
          f"{need / gl5['evals']:.0f}x the model evaluations.")
    out = os.path.join(HERE, "exposure.json")
    with open(out, "w") as f:
        json.dump({"reference_agreement": agree, "rows": rows}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

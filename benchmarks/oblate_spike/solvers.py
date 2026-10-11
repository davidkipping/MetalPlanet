"""Quartic solvers for D(z) = S2 z^4 + S1 z^3 + (S0-p) z^2 + conj(S1) z + S2
(self-inversive), in fp32 vs complex128 eigenvalues as truth."""
import numpy as np
c64, c128 = np.complex64, np.complex128
d = np.load(sys_argv := __import__("sys").argv[1])
C = d["C"]; eps = d["eps"]
x0, y0, A, B = C[:4]; f = C[5]
keep = f >= 1e-5                                   # the regime the oblate kernel serves in fp32
x0, y0, A, B = x0[keep], y0[keep], A[keep], B[keep]

def coeffs(x0, y0, a, b, p, dt):
    S0 = (x0 * x0 + y0 * y0 + 0.5 * (a * a + b * b)).astype(dt)
    S1 = (a * x0 - 1j * (b * y0)).astype(dt)
    S2 = (0.25 * (a * a - b * b)).astype(dt)
    return [S2, S1, (S0 - p).astype(dt), np.conj(S1), S2]     # high -> low

def D(z, k):  return (((k[0] * z + k[1]) * z + k[2]) * z + k[3]) * z + k[4]
def dD(z, k): return ((4 * k[0] * z + 3 * k[1]) * z + 2 * k[2]) * z + k[3]

def eig_roots(k, dt):
    n = k[0].size; M = np.zeros((n, 4, 4), dtype=dt)
    for j in range(4): M[:, 0, j] = -k[j + 1] / k[0]
    M[:, 1, 0] = M[:, 2, 1] = M[:, 3, 2] = 1
    return np.linalg.eigvals(M).astype(dt)

def seeds(k):
    """circle-limit seeds: two finite roots of S1 z^2 + (S0-p) z + conj S1,
    the small root ~ -S2 / conj S1, the large one its inverse conjugate."""
    a2, a1, a0 = k[1], k[2], k[3]
    disc = np.sqrt(a1 * a1 - 4 * a2 * a0)
    q = -0.5 * (a1 + np.where(np.real(np.conj(a1) * disc) >= 0, disc, -disc))
    small = -k[0] / k[3]
    return np.stack([q / a2, a0 / q, small, 1 / np.conj(small)], 1)

def aberth(k, z, iters):
    for _ in range(iters):
        r = D(z, [c[:, None] for c in k]) / dD(z, [c[:, None] for c in k])
        s = np.zeros_like(z)
        for i in range(4):
            for j in range(4):
                if i != j: s[:, i] += 1 / (z[:, i] - z[:, j])
        z = z - r / (1 - r * s)
    return z

def polish(k, z, iters=2):
    kk = [c[:, None] for c in k]
    for _ in range(iters):
        z = z - D(z, kk) / dD(z, kk)
    return z

def self_inversive_newton(k, iters):
    """Only two roots: Newton from the two seeds inside the unit disc, the
    other two by z -> 1/conj(z)."""
    s = seeds(k)
    inner = np.where(np.abs(s[:, :2]) <= 1, s[:, :2], 1 / np.conj(s[:, :2]))
    z = np.stack([inner[:, 0], s[:, 2]], 1)
    kk = [c[:, None] for c in k]
    for _ in range(iters):
        z = z - D(z, kk) / dD(z, kk)
    return np.concatenate([z, 1 / np.conj(z)], 1)

def ferrari(k):
    a, b, c, dd = (kk / k[0] for kk in k[1:])
    p = b - 3 * a * a / 8; q = c - a * b / 2 + a ** 3 / 8; r = dd - a * c / 4 + a * a * b / 16 - 3 * a ** 4 / 256
    # resolvent m^3 + p m^2 + (p^2/4 - r) m - q^2/8 = 0  (Cardano)
    B2, B1, B0 = p, p * p / 4 - r, -q * q / 8
    P = B1 - B2 * B2 / 3; Q = 2 * B2 ** 3 / 27 - B2 * B1 / 3 + B0
    sq = np.sqrt(Q * Q / 4 + P ** 3 / 27)
    u = (-Q / 2 + np.where(np.abs(-Q / 2 + sq) >= np.abs(-Q / 2 - sq), sq, -sq)) ** (1 / 3)
    m = u - P / (3 * np.where(np.abs(u) > 0, u, 1)) - B2 / 3
    s2m = np.sqrt(2 * m)
    t = q / (np.where(np.abs(s2m) > 0, s2m, 1))
    r1 = np.sqrt(-(2 * p + 2 * m) - 2 * t); r2 = np.sqrt(-(2 * p + 2 * m) + 2 * t)
    y = np.stack([(s2m + r1) / 2, (s2m - r1) / 2, (-s2m + r2) / 2, (-s2m - r2) / 2], 1)
    return y - a[:, None] / 4

def match_err(est, ref):
    """relative error after optimal matching (greedy on the exact roots)"""
    est = est.astype(c128); err = np.zeros(ref.shape)
    used = np.zeros(est.shape, bool)
    for i in range(4):
        dist = np.abs(est - ref[:, i:i + 1]); dist[used] = np.inf
        j = np.argmin(dist, 1); used[np.arange(len(j)), j] = True
        err[:, i] = dist[np.arange(len(j)), j] / np.maximum(1.0, np.abs(ref[:, i]))
    return err

print(f"{len(x0)} configurations with f >= 1e-5 (realistic chords; r 0.01-0.3)")
print(f"{'level':>10s} {'solver':>26s} {'worst rel err':>14s} {'p99':>10s} {'near-circle roots worst':>24s} {'pair (sig,pi) worst':>20s}")
for p in [1.0] + [1 + e for e in eps]:
    k128 = coeffs(x0, y0, A, B, p, c128); ref = eig_roots(k128, c128); ref = polish(k128, ref, 3)
    k32 = coeffs(x0, y0, A, B, p, c64)
    near = np.abs(np.abs(ref) - 1) < 1e-2
    # closest pair (the "near-double" candidate) in the truth
    dd = np.abs(ref[:, :, None] - ref[:, None, :]) + np.eye(4)[None] * 1e9
    ii, jj = np.unravel_index(dd.reshape(len(ref), -1).argmin(1), (4, 4))
    rows = np.arange(len(ref)); sig_t, pi_t = ref[rows, ii] + ref[rows, jj], ref[rows, ii] * ref[rows, jj]
    for name, z in (("LAPACK eig (fp32)", polish(k32, eig_roots(k32, c64))),
                    ("Ferrari + 2 Newton", polish(k32, ferrari(k32).astype(c64))),
                    ("Aberth x6 (circle seeds)", polish(k32, aberth(k32, seeds(k32).astype(c64), 6))),
                    ("self-inversive Newton x6", self_inversive_newton(k32, 6))):
        e = match_err(z, ref)
        z128 = z.astype(c128)
        # pair from the estimate: the two estimated roots nearest the true pair
        zi = z128[rows, np.abs(z128 - ref[rows, ii][:, None]).argmin(1)]; zj = z128[rows, np.abs(z128 - ref[rows, jj][:, None]).argmin(1)]
        pe = np.maximum(np.abs(zi + zj - sig_t) / np.maximum(1, np.abs(sig_t)), np.abs(zi * zj - pi_t) / np.maximum(1, np.abs(pi_t)))
        e = np.where(np.isfinite(e), e, 9.9)
        print(f"{('limb' if p == 1.0 else f'p=1+{p-1:.4g}'):>10s} {name:>26s} {e.max():14.1e} {np.percentile(e.max(1), 99):10.1e} {e[near].max() if near.any() else 0:24.1e} {np.nanmax(pe):20.1e}")

"""Foster and Cauer thermal networks: conversion, checks and response to a power waveform.

Companion module of foster_to_cauer.tex (research/power_semiconductor_devices/thermal/foster_to_cauer).
Units: R in K/W, tau in s, C in J/K, p in W, temperatures in K or degC (rises in K).

Foster network, the form of the datasheets:
    Z(s) = sum_i R_i / (1 + s tau_i),            Z_th(t) = sum_i R_i (1 - exp(-t / tau_i))
Cauer network (ladder): C_1 at the junction node, then R_1, C_2, R_2, ..., C_n, and R_n to the case:
    Z(s) = 1 / (s C_1 + 1 / (R_1 + 1 / (s C_2 + 1 / (R_2 + ... + 1 / (s C_n + 1 / R_n)))))

Conversions
    foster_to_cauer(R, tau)          continued-fraction expansion of Y = 1/Z with numpy polynomials; the function
                                     of the thermal check of a press-pack crowbar stack, unchanged in its steps
    foster_to_cauer_mp(R, tau)       the same expansion with 120 significant digits (reference for the checks)
    foster_to_cauer_bidiag(R, tau)   the same ladder from the eigenvalues and residues of Z(s), by Golub-Kahan
                                     bidiagonalization: no polynomials, only products and quotients of positive
                                     numbers, stable for many terms and widely spread time constants
    cauer_to_foster(R, C)            the inverse transformation, from the eigenvalues of the ladder
Other functions
    zth_foster, zth_cauer            step responses Z_th(t)
    invariants(R, tau)               quantities that every exact conversion preserves
    split_sides(R, C, R_sides)       per-side ladders of a double-side cooled device
    structure_function(R, C)         cumulative thermal capacitance against cumulative thermal resistance
    Network                          node network (ladders, contacts, heat sinks) and its exact response to a
                                     piecewise-constant power

Run as a program, the module checks the three conversions against each other and against the round trip.
Only numpy is needed (scipy for nothing here).
"""
import decimal

import numpy as np


# ------------------------------------------------------------------------------------------------ conversions
def foster_to_cauer(R, tau):
    """Cauer ladder (C at the junction node, then R, C, ...; the last R goes to the case) with the same Z_th(s).

    Polynomials in descending powers of s (numpy convention). Z(s) = N(s)/D(s) with
    D = prod_i (tau_i s + 1) and N = sum_k R_k prod_{j != k} (tau_j s + 1); the admittance Y = D/N has a
    numerator one degree higher than its denominator. Each pass of the loop removes from the current admittance
    its pole at infinity, s C (C = ratio of the leading coefficients), inverts the remainder, removes from the
    impedance its value at infinity, R (again a ratio of leading coefficients), and inverts again.
    """
    n = len(R)
    D = np.array([1.0])
    for ti in tau:
        D = np.polymul(D, [ti, 1.0])
    N = np.zeros(1)
    for k in range(n):
        term = np.array([R[k]])
        for j in range(n):
            if j != k:
                term = np.polymul(term, [tau[j], 1.0])
        N = np.polyadd(N, term)
    num, den = D, N                      # Y(s) = num / den, deg(num) = deg(den) + 1
    Cs, Rs = [], []
    for _ in range(n):
        C = num[0] / den[0]
        rem = np.polysub(num, np.polymul([C, 0.0], den))[1:]      # Y - s C   (leading term cancels)
        Rk = den[0] / rem[0]
        rem2 = np.polysub(den, Rk * rem)[1:]                       # Z - R     (leading term cancels)
        Cs.append(C); Rs.append(Rk)
        num, den = rem, rem2
    return np.array(Rs), np.array(Cs)


def foster_to_cauer_mp(R, tau, digits=120):
    """foster_to_cauer carried out with 'digits' significant digits (Python decimal) instead of double precision:
    the reference of the accuracy checks. Every float of the input is taken at its exact binary value."""
    ctx = decimal.Context(prec=digits)
    dec = lambda x: ctx.create_decimal(float(x))
    zero, one = ctx.create_decimal(0), ctx.create_decimal(1)

    def padd(a, b, sign=1):
        n = max(len(a), len(b))
        a = [zero] * (n - len(a)) + list(a)
        b = [zero] * (n - len(b)) + list(b)
        return [ctx.add(x, y) if sign > 0 else ctx.subtract(x, y) for x, y in zip(a, b)]

    def pmul(a, b):
        out = [zero] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i + j] = ctx.add(out[i + j], ctx.multiply(x, y))
        return out

    R = [dec(x) for x in R]
    tau = [dec(x) for x in tau]
    n = len(R)
    D = [one]
    for ti in tau:
        D = pmul(D, [ti, one])
    N = [zero]
    for k in range(n):
        term = [R[k]]
        for j in range(n):
            if j != k:
                term = pmul(term, [tau[j], one])
        N = padd(N, term)
    num, den = D, N
    Cs, Rs = [], []
    for _ in range(n):
        C = ctx.divide(num[0], den[0])
        rem = padd(num, pmul([C, zero], den), -1)[1:]                 # Y - s C
        Rk = ctx.divide(den[0], rem[0])
        rem2 = padd(den, [ctx.multiply(Rk, x) for x in rem], -1)[1:]    # Z - R
        Cs.append(C); Rs.append(Rk)
        num, den = rem, rem2
    return np.array([float(x) for x in Rs]), np.array([float(x) for x in Cs])


def foster_to_cauer_bidiag(R, tau):
    """Cauer ladder from the eigenvalue form of Z(s), by Golub-Kahan bidiagonalization.

    Z(s) = sum_i w_i / (s + lam_i) with lam_i = 1/tau_i and w_i = R_i/tau_i. For the ladder,
    M = C^-1/2 G C^-1/2 = B B^T with B lower bidiagonal, B_kk = 1/sqrt(C_k R_k), B_k+1,k = -1/sqrt(C_k+1 R_k);
    the eigenvalues of M are lam_i and the first components of its unit eigenvectors are u_i = sqrt(w_i C_1),
    with C_1 = 1/sum(w). Bidiagonalizing A = diag(sqrt(lam)) with the starting vector u gives B (up to signs),
    and the ladder follows with products and quotients only.
    """
    R = np.asarray(R, float)
    tau = np.asarray(tau, float)
    n = len(R)
    lam = 1.0 / tau
    w = R / tau
    C1 = 1.0 / w.sum()
    a = np.sqrt(lam)
    U = np.zeros((n, n))
    V = np.zeros((n, n))
    alpha = np.zeros(n)
    beta = np.zeros(n)                   # beta[k] couples step k-1 and step k (beta[0] unused)
    U[:, 0] = np.sqrt(w * C1)
    U[:, 0] /= np.linalg.norm(U[:, 0])
    v = a * U[:, 0]
    alpha[0] = np.linalg.norm(v)
    V[:, 0] = v / alpha[0]
    for k in range(1, n):
        x = a * V[:, k - 1] - alpha[k - 1] * U[:, k - 1]
        for _ in range(2):               # full reorthogonalization, twice
            x -= U[:, :k] @ (U[:, :k].T @ x)
        beta[k] = np.linalg.norm(x)
        U[:, k] = x / beta[k]
        y = a * U[:, k] - beta[k] * V[:, k - 1]
        for _ in range(2):
            y -= V[:, :k] @ (V[:, :k].T @ y)
        alpha[k] = np.linalg.norm(y)
        V[:, k] = y / alpha[k]
    Cs, Rs = np.zeros(n), np.zeros(n)
    Cs[0] = C1
    for k in range(n):
        Rs[k] = 1.0 / (alpha[k] ** 2 * Cs[k])
        if k + 1 < n:
            Cs[k + 1] = 1.0 / (beta[k + 1] ** 2 * Rs[k])
    return Rs, Cs


def ladder_matrix(R, C):
    """Symmetric matrix M = C^-1/2 G C^-1/2 of a Cauer ladder whose last resistance goes to a fixed temperature."""
    R = np.asarray(R, float)
    C = np.asarray(C, float)
    n = len(R)
    G = np.zeros((n, n))
    for k in range(n):
        G[k, k] += 1.0 / R[k]
        if k + 1 < n:
            G[k + 1, k + 1] += 1.0 / R[k]
            G[k, k + 1] -= 1.0 / R[k]
            G[k + 1, k] -= 1.0 / R[k]
    s = 1.0 / np.sqrt(C)
    return s[:, None] * G * s[None, :]


def cauer_to_foster(R, C):
    """Foster terms (R_i, tau_i), longest time constant first, of the ladder (R, C)."""
    lam, Q = np.linalg.eigh(ladder_matrix(R, C))
    Rf = Q[0, :] ** 2 / (C[0] * lam)
    order = np.argsort(lam)
    return Rf[order], 1.0 / lam[order]


# ------------------------------------------------------------------------------------------------ responses, checks
def zth_foster(R, tau, t):
    t = np.asarray(t, float)[..., None]
    return np.sum(np.asarray(R) * -np.expm1(-t / np.asarray(tau)), axis=-1)


def zth_cauer(R, C, t):
    Rf, tf = cauer_to_foster(R, C)
    return zth_foster(Rf, tf, t)


def invariants(R, tau):
    """Quantities fixed by Z(s) alone: total resistance, junction capacitance C_1 = 1/sum(R/tau) (initial slope
    of Z_th), and the first moment sum(R tau) = sum_k C_k theta_k^2 of the ladder (theta_k = resistance from node
    k to the case)."""
    R = np.asarray(R, float)
    tau = np.asarray(tau, float)
    return dict(R_tot=R.sum(), C1=1.0 / np.sum(R / tau), m1=np.sum(R * tau))


def ladder_moment(R, C):
    theta = np.cumsum(np.asarray(R)[::-1])[::-1]       # resistance from node k to the case
    return np.sum(np.asarray(C) * theta ** 2)


def split_sides(R, C, R_sides):
    """Per-side ladders of a double-side cooled device: same ladder with R x k and C / k on each side, the
    junction capacitance C_1 shared. k_side = R_side / R_par, R_par = parallel of R_sides, so that sum(1/k) = 1:
    the two ladders in parallel give back Z(s) exactly, and the total capacitance is preserved."""
    R_par = 1.0 / sum(1.0 / r for r in R_sides)
    ks = [r / R_par for r in R_sides]
    return ks, [(np.asarray(R, float) * k, np.asarray(C[1:], float) / k) for k in ks]


def structure_function(R, C):
    """Staircase of the cumulative structure function: points (R_sum, C_sum) from the junction to the case."""
    Rs = np.concatenate([[0.0], np.cumsum(R)])
    Cs = np.cumsum(C)
    x, y = [], []
    for k in range(len(R)):
        x += [Rs[k], Rs[k + 1]]
        y += [Cs[k], Cs[k]]
    return np.array(x), np.array(y)


# ------------------------------------------------------------------------------------------------ networks
class Network:
    """Thermal network of nodes with capacitance (possibly zero) and conductances; 'ground' is the fixed
    reference temperature (ideal heat sink or ambient). Temperatures are rises over that reference."""

    def __init__(self):
        self.C = []
        self.edges = []
        self.g_ref = {}

    def node(self, C=0.0):
        self.C.append(float(C))
        return len(self.C) - 1

    def link(self, a, b, R):
        self.edges.append((a, b, 1.0 / R))

    def ground(self, a, R):
        self.g_ref[a] = self.g_ref.get(a, 0.0) + 1.0 / R

    def ladder(self, start, R, C_inner, end=None):
        """Ladder from node 'start': R[0], node C_inner[0], R[1], ..., R[-1] to 'end' (a new massless node if
        None). Returns the end node."""
        prev = start
        for k in range(len(R) - 1):
            nk = self.node(C_inner[k])
            self.link(prev, nk, R[k])
            prev = nk
        if end is None:
            end = self.node(0.0)
        self.link(prev, end, R[-1])
        return end

    def matrices(self):
        n = len(self.C)
        G = np.zeros((n, n))
        for a, b, g in self.edges:
            G[a, a] += g; G[b, b] += g; G[a, b] -= g; G[b, a] -= g
        for a, g in self.g_ref.items():
            G[a, a] += g
        return np.array(self.C), G

    def modes(self, b):
        """Modal form of Cm dT/dt = -G T + b p after removing the nodes without capacitance."""
        Cv, G = self.matrices()
        n = len(Cv)
        dyn = np.where(Cv > 0)[0]
        alg = np.setdiff1d(np.arange(n), dyn)
        b = np.asarray(b, float)
        if len(alg):
            Gaa_inv = np.linalg.inv(G[np.ix_(alg, alg)])
            Gr = G[np.ix_(dyn, dyn)] - G[np.ix_(dyn, alg)] @ Gaa_inv @ G[np.ix_(alg, dyn)]
            br = b[dyn] - G[np.ix_(dyn, alg)] @ Gaa_inv @ b[alg]
        else:
            Gaa_inv = None
            Gr, br = G, b
        s = 1.0 / np.sqrt(Cv[dyn])
        Msym = s[:, None] * Gr * s[None, :]
        Msym = 0.5 * (Msym + Msym.T)
        lam, Q = np.linalg.eigh(Msym)
        lam = np.where(np.abs(lam) < 1e-12 * max(1.0, np.abs(lam).max()), 0.0, lam)
        P = s[:, None] * Q                              # T_dyn = P x
        f = Q.T @ (s * br)                              # dx/dt = -lam x + f p
        return dict(dyn=dyn, alg=alg, Gaa_inv=Gaa_inv, G=G, b=b, lam=lam, P=P, f=f, n=n)

    def response(self, b, t_edges, p_steps, out):
        """Temperature rises at the instants t_edges (first = 0) for the power p_steps[k] held constant on
        (t_edges[k], t_edges[k+1]); exact for piecewise-constant power. b: injection weights (sum 1);
        out: list of readout weight vectors. Returns an array (len(t_edges), len(out)); a node without
        capacitance is read with the power of the interval that ends at that instant."""
        m = self.modes(b)
        lam, P, f = m["lam"], m["P"], m["f"]
        x = np.zeros(len(lam))
        W = np.array(out, float)
        res = np.zeros((len(t_edges), len(out)))
        for k in range(len(t_edges) - 1):
            dt = t_edges[k + 1] - t_edges[k]
            e = np.exp(-lam * dt)
            phi = np.where(lam > 0, -np.expm1(-lam * dt) / np.where(lam > 0, lam, 1.0), dt)
            x = e * x + phi * f * p_steps[k]
            res[k + 1] = W @ self._full(m, P @ x, p_steps[k])
        return res

    def _full(self, m, Tdyn, p):
        T = np.zeros(m["n"])
        T[m["dyn"]] = Tdyn
        if len(m["alg"]):
            G, alg, dyn = m["G"], m["alg"], m["dyn"]
            T[alg] = m["Gaa_inv"] @ (m["b"][alg] * p - G[np.ix_(alg, dyn)] @ Tdyn)
        return T


def foster_response(R, tau, t_edges, p_steps):
    """Temperature drop across a Foster chain (junction minus case) for a piecewise-constant power: every cell
    carries the whole power, so the drop does not depend on what is connected to the case."""
    R = np.asarray(R, float)
    tau = np.asarray(tau, float)
    th = np.zeros(len(R))
    out = np.zeros(len(t_edges))
    for k in range(len(t_edges) - 1):
        e = np.exp(-(t_edges[k + 1] - t_edges[k]) / tau)
        th = e * th + R * (1.0 - e) * p_steps[k]
        out[k + 1] = th.sum()
    return out


# ------------------------------------------------------------------------------------------------ self-check
def _check():
    cases = {
        "4 terms, 0.5 ms - 0.43 s": ([3.587e-3, 0.838e-3, 0.505e-3, 0.069e-3], [0.4299, 0.0405, 0.0028, 0.0005]),
        "4 terms, 2.2 ms - 0.44 s": ([4.533e-3, 2.255e-3, 0.868e-3, 0.345e-3], [0.4406, 0.1045, 0.0092, 0.0022]),
        "8 terms, 1 us - 10 s": (list(np.full(8, 1e-3)), list(np.logspace(1, -6, 8))),
    }
    for name, (R, tau) in cases.items():
        Re, Ce = foster_to_cauer_mp(R, tau)
        for f in (foster_to_cauer, foster_to_cauer_bidiag):
            Rc, Cc = f(R, tau)
            err = max(np.max(np.abs(Rc / Re - 1)), np.max(np.abs(Cc / Ce - 1)))
            print(f"{name:26s} {f.__name__:24s} max relative error of the elements {err:.1e}")
        Rf, tf = cauer_to_foster(Re, Ce)
        print(f"{'':26s} round trip: R {np.max(np.abs(np.sort(Rf) / np.sort(R) - 1)):.1e}, "
              f"tau {np.max(np.abs(np.sort(tf) / np.sort(tau) - 1)):.1e}")


if __name__ == "__main__":
    _check()

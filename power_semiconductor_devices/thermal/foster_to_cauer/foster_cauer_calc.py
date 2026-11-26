"""Worked examples of foster_to_cauer.tex: from the Foster network of a datasheet to the Cauer ladder, and what the
ladder is for.

The script applies the module foster_cauer.py and writes every number quoted in the note:
  1. two press-pack devices with the Foster networks of their datasheets (double-side cooled): rectifier diode
     5SDD 70H2000 and phase-control thyristor 5STP 50Q1800. Cauer ladder, invariants, Foster capacitances,
     per-side ladders, check of the step response;
  2. the diode clamped between two copper bars (lumped, adiabatic) through a contact resistance, under a
     rectangular power pulse: Cauer per side against the Foster network attached to the bars;
  3. a synthetic press-pack whose layers are known (1D finite-difference model, Si between Mo discs and Cu poles):
     its exact Z_th, Foster fits of 3 to 6 terms on 1 ms - 10 s as a datasheet would give, their Cauer ladders,
     the structure function, the per-side split against the single-side curves of the model, and the
     junction temperature on the bars against the model itself;
  4. accuracy of the polynomial expansion and of the bidiagonalization against a 200-digit reference, for
     2 to 60 terms over 1 to 8 decades.

Outputs, next to the script: foster_cauer_results.txt (log), foster_cauer_values.tex (macros \\fc... of the note,
read by both language versions), figures/fig_*.pdf and figures/fig_*_it.pdf (the same figures with Italian labels,
for foster_to_cauer_it.tex). numpy, scipy and matplotlib (numpy 1.x and 2.x). About ten seconds.
Run:  python foster_cauer_calc.py"""
import math
import os
import time
import warnings
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import foster_cauer as fc

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- data of the worked examples
# Datasheet Foster networks, junction to case, double-side cooled (R in K/W, tau in s)
DIODE = dict(name="5SDD 70H2000",
             R=[4.533e-3, 2.255e-3, 0.868e-3, 0.345e-3], tau=[0.4406, 0.1045, 0.0092, 0.0022],
             R_sides=(14.5e-3, 18.0e-3))          # datasheet: anode-side and cathode-side cooled
THYR = dict(name="5STP 50Q1800",
            R=[3.587e-3, 0.838e-3, 0.505e-3, 0.069e-3], tau=[0.4299, 0.0405, 0.0028, 0.0005],
            R_sides=(10.0e-3, 10.0e-3))           # assumption: equal sides (no single-side values used)
# External network of the coupling examples, per side: contact case-bar and copper bar, lumped and adiabatic
R_CH = 5.0e-3                                     # K/W per side
C_BAR = 620.0                                     # J/K per side: Cu bar 100 x 10 x 180 mm
P_PULSE, T_PULSE = 15e3, 0.5                      # W, s: rectangular pulse of the time-domain example
# Synthetic press-pack: area of a 63 mm pole, layers from the anode case to the cathode case
D_POLE = 0.063
MAT = dict(Cu=(390.0, 8930.0 * 385.0), Mo=(138.0, 10220.0 * 251.0), Si=(120.0, 2330.0 * 705.0))   # k, rho c
# a layer is (material, thickness m, cells); a contact is ("contact", resistance K/W)
LAYERS = [("Cu", 8.0e-3, 32), ("contact", 2.5e-3), ("Mo", 2.0e-3, 20), ("Si", 0.5e-3, 10),     # sides differ
          ("contact", 4.0e-3), ("Mo", 1.5e-3, 15), ("contact", 2.5e-3), ("Cu", 8.0e-3, 32)]
LAYERS_SYM = [("Cu", 8.0e-3, 32), ("contact", 2.5e-3), ("Mo", 2.0e-3, 20), ("Si", 0.5e-3, 10),  # equal sides
              ("Mo", 2.0e-3, 20), ("contact", 2.5e-3), ("Cu", 8.0e-3, 32)]
T_FIT = (1e-3, 10.0)                              # s: time window of the "datasheet" curve of the synthetic device
N_FITS = (3, 4, 5, 6)
# accuracy study
STUDY_N = list(range(2, 61, 2))
STUDY_DEC = (1, 2, 4, 8)
MP_DIGITS = 200

# ---------------------------------------------------------------- style of the figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_A, C_B, C_C, C_D = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"       # categorical colours, fixed order
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]                  # ordinal ramp, light to dark
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

LOG, MAC = [], {}


def line(s=""):
    print(s)
    LOG.append(s)


def rnd(x, nd):
    """Round half up (as written by hand), returned as a string with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def mac(name, value, nd=None):
    MAC[name] = rnd(value, nd) if nd is not None else str(value)


def sci_tex(x, nd=0):
    """x as m \\times 10^{e} for the note."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if float(rnd(m, nd)) >= 10:
        e += 1
        m = x / 10 ** e
    return f"{rnd(m, nd)}\\times 10^{{{e}}}"


# ---------------------------------------------------------------- 1. datasheet networks
def device(dv, tag):
    R, tau = np.array(dv["R"]), np.array(dv["tau"])
    Rc, Cc = fc.foster_to_cauer(R, tau)
    Rm, Cm = fc.foster_to_cauer_mp(R, tau, MP_DIGITS)
    Rb, Cb = fc.foster_to_cauer_bidiag(R, tau)
    err_poly = max(np.max(np.abs(Rc / Rm - 1)), np.max(np.abs(Cc / Cm - 1)))
    err_bid = max(np.max(np.abs(Rb / Rm - 1)), np.max(np.abs(Cb / Cm - 1)))
    Rf, tf = fc.cauer_to_foster(Rc, Cc)
    err_rt = max(np.max(np.abs(Rf / R - 1)), np.max(np.abs(tf / tau - 1)))
    t = np.logspace(-6, 3, 901)
    zf, zc = fc.zth_foster(R, tau, t), fc.zth_cauer(Rc, Cc, t)
    err_z = np.max(np.abs(zc / zf - 1))
    inv = fc.invariants(R, tau)
    m1c = fc.ladder_moment(Rc, Cc)
    Cfos = tau / R
    theta = np.cumsum(Rc[::-1])[::-1]                  # resistance from node k to the case
    delay = np.sum(Cc * theta)                         # mean delay of the heat leaving the case (s)
    a1, a2 = np.sum(R / tau), np.sum(R / tau ** 2)
    ks, sides = fc.split_sides(Rc, Cc, dv["R_sides"])
    line(f"{dv['name']}: Foster R {np.round(R * 1e3, 3)} K/kW, tau {tau} s, R_th(j-c) {R.sum() * 1e3:.3f} K/kW")
    line(f"   Cauer R {np.round(Rc * 1e3, 4)} K/kW, C {np.round(Cc, 3)} J/K; sum C {Cc.sum():.2f} J/K, "
         f"R_k C_k {np.round(Rc * Cc * 1e3, 3)} ms")
    line(f"   Foster capacitances tau/R {np.round(Cfos, 2)} J/K, sum {Cfos.sum():.1f} J/K, series {1 / np.sum(1 / Cfos):.3f} J/K")
    line(f"   invariants: sum R {inv['R_tot'] * 1e3:.4f} / {Rc.sum() * 1e3:.4f} K/kW; C_1 = 1/sum(R/tau) {inv['C1']:.4f} "
         f"/ {Cc[0]:.4f} J/K; sum R tau {inv['m1'] * 1e3:.5f} / sum C theta^2 {m1c * 1e3:.5f} K s/kW")
    line(f"   first section in closed form: C_1 = 1/sum(R/tau) {1 / a1:.4f} J/K, R_1 = sum(R/tau)^2/sum(R/tau^2) "
         f"{a1 ** 2 / a2 * 1e3:.4f} K/kW; stored heat at 1 W, sum C_k theta_k = mean delay of the heat {delay:.3f} s")
    line(f"   accuracy: polynomial vs {MP_DIGITS} digits {err_poly:.1e}, bidiagonalization {err_bid:.1e}, "
         f"round trip {err_rt:.1e}, Z_th(t) 1 us - 1000 s {err_z:.1e}")
    R_par = 1 / sum(1 / r for r in dv["R_sides"])
    for (Rs, Cs), k, rs in zip(sides, ks, dv["R_sides"]):
        line(f"   side R_th {rs * 1e3:.1f} K/kW: k = {k:.4f}, ladder R {np.round(Rs * 1e3, 3)} K/kW (sum {Rs.sum() * 1e3:.3f}), "
             f"C (after the junction) {np.round(Cs, 2)} J/K")
    line(f"   parallel of the side values {R_par * 1e3:.3f} K/kW; share of the heat to the first side "
         f"{dv['R_sides'][1] / sum(dv['R_sides']) * 100:.1f} %")
    for i in range(4):
        mac(f"{tag}R{'ABCD'[i]}", R[i] * 1e3, 3)
        mac(f"{tag}T{'ABCD'[i]}", tau[i], 4)
        mac(f"{tag}Rc{'ABCD'[i]}", Rc[i] * 1e3, 3)
        mac(f"{tag}Cc{'ABCD'[i]}", Cc[i], 2 if Cc[i] < 10 else 1)
        mac(f"{tag}Cf{'ABCD'[i]}", Cfos[i], 1 if Cfos[i] >= 10 else 2)
        mac(f"{tag}RCc{'ABCD'[i]}", Rc[i] * Cc[i] * 1e3, 2)
    mac(f"{tag}Rth", R.sum() * 1e3, 3)
    mac(f"{tag}Delay", delay, 2)
    mac(f"{tag}Stored", delay * 1e3, 0)
    for i in range(4):
        mac(f"{tag}Theta{'ABCD'[i]}", theta[i] * 1e3, 3)
    mac(f"{tag}CcSum", Cc.sum(), 1)
    mac(f"{tag}CfSum", Cfos.sum(), 1)
    mac(f"{tag}CfSeries", 1 / np.sum(1 / Cfos), 2)
    mac(f"{tag}Mone", inv["m1"] * 1e3, 4)
    mac(f"{tag}ErrPoly", sci_tex(max(err_poly, 1e-17)), None)
    mac(f"{tag}ErrBid", sci_tex(max(err_bid, 1e-17)), None)
    mac(f"{tag}ErrRt", sci_tex(max(err_rt, 1e-17)), None)
    mac(f"{tag}ErrZ", sci_tex(max(err_z, 1e-17)), None)
    mac(f"{tag}Rpar", R_par * 1e3, 2)
    for j, (k, (Rs, Cs)) in enumerate(zip(ks, sides)):
        s = "ab"[j]
        mac(f"{tag}K{s}", k, 3)
        mac(f"{tag}RsSum{s}", Rs.sum() * 1e3, 2)
        for i in range(4):
            mac(f"{tag}Rs{s}{'ABCD'[i]}", Rs[i] * 1e3, 3)
        for i in range(3):
            mac(f"{tag}Cs{s}{'BCD'[i]}", Cs[i], 2 if Cs[i] < 10 else 1)
    mac(f"{tag}ShareA", dv["R_sides"][1] / sum(dv["R_sides"]) * 100, 1)
    return dict(R=R, tau=tau, Rc=Rc, Cc=Cc, ks=ks, sides=sides, Cfos=Cfos, t=t, zf=zf)


# ---------------------------------------------------------------- 2. coupling to the bars
def coupled_cauer(Rc, Cc, ks, with_bars=True):
    """Junction node with C_1, one ladder per side, case nodes, contact and bar per side (or ideal sink)."""
    net = fc.Network()
    j = net.node(Cc[0])
    cases, bars = [], []
    for k in ks:
        Rs, Cs = np.asarray(Rc) * k, np.asarray(Cc[1:]) / k
        if with_bars:
            c = net.ladder(j, Rs, Cs)
            b = net.node(C_BAR)
            net.link(c, b, R_CH)
            cases.append(c); bars.append(b)
        else:
            prev = j
            for r, cap in zip(Rs[:-1], Cs):
                nk = net.node(cap); net.link(prev, nk, r); prev = nk
            net.ground(prev, Rs[-1])
            cases.append(None)
    return net, j, cases, bars


def bars_common_case():
    """External network seen by a Foster chain: one case node, contact and bar on each side."""
    net = fc.Network()
    c = net.node(0.0)
    for _ in range(2):
        b = net.node(C_BAR)
        net.link(c, b, R_CH)
    return net, c


def unit(n, i):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def step_response(net, b, t, out):
    """Step response (power 1 W from t = 0) at the instants t (exact)."""
    m = net.modes(b)
    lam, P, f = m["lam"], m["P"], m["f"]
    W = np.array(out, float)
    res = np.zeros((len(t), len(out)))
    for i, ti in enumerate(t):
        phi = np.where(lam > 0, -np.expm1(-lam * ti) / np.where(lam > 0, lam, 1.0), ti)
        res[i] = W @ net._full(m, P @ (phi * f), 1.0)
    return res


def coupling_example(d):
    Rc, Cc, ks = d["Rc"], d["Cc"], d["ks"]
    R, tau = d["R"], d["tau"]
    # time grid: pulse of T_PULSE, observed up to 2 s
    t_edges = np.unique(np.concatenate([np.linspace(0, T_PULSE, 2001), np.linspace(T_PULSE, 2.0, 1501)]))
    p_steps = np.where(t_edges[:-1] < T_PULSE - 1e-12, P_PULSE, 0.0)
    # Cauer per side on the bars: junction, cases, heat into each bar
    net, j, cases, bars = coupled_cauer(Rc, Cc, ks)
    n = len(net.C)
    q_out = [(unit(n, c) - unit(n, b)) / R_CH for c, b in zip(cases, bars)]
    resC = net.response(unit(n, j), t_edges, p_steps, [unit(n, j), unit(n, cases[0]), unit(n, cases[1])] + q_out)
    # Foster on the bars, one case node
    netF, cF = bars_common_case()
    resF_case = netF.response(unit(len(netF.C), cF), t_edges, p_steps, [unit(len(netF.C), cF)])[:, 0]
    resF = fc.foster_response(R, tau, t_edges, p_steps) + resF_case
    # ideal heat sink
    resI = fc.foster_response(R, tau, t_edges, p_steps)
    k_end = np.searchsorted(t_edges, T_PULSE)
    out = dict(t=t_edges, TjC=resC[:, 0], TcC=resC[:, 1:3], qC=resC[:, 3:5], TjF=resF, TcF=resF_case, TjI=resI)
    TjC, TjF, TjI = resC[k_end, 0], resF[k_end], resI[k_end]
    line(f"   {P_PULSE / 1e3:.0f} kW for {T_PULSE:.1f} s, bars {C_BAR:.0f} J/K and contact {R_CH * 1e3:.0f} K/kW per side:")
    line(f"      junction rise at the end of the pulse: ideal sink {TjI:.1f} K, Cauer per side on the bars {TjC:.1f} K, "
         f"Foster on the bars {TjF:.1f} K (+{TjF - TjC:.1f} K)")
    line(f"      cases at the end: Cauer {resC[k_end, 1]:.1f} / {resC[k_end, 2]:.1f} K, Foster {resF_case[k_end]:.1f} K; "
         f"bars: {np.mean([resC[k_end, 1] - resC[k_end, 3] * R_CH, resC[k_end, 2] - resC[k_end, 4] * R_CH]):.1f} K")
    qtot = resC[:, 3] + resC[:, 4]
    k10 = int(np.argmax(qtot >= 0.1 * P_PULSE))
    line(f"      heat into the bars (Cauer): 10 % of the power after {t_edges[k10] * 1e3:.0f} ms, "
         f"{qtot[k_end] / P_PULSE * 100:.0f} % at the end of the pulse; with the Foster network 100 % from the start")
    E_stored_end = P_PULSE * T_PULSE - np.trapezoid(qtot[: k_end + 1], t_edges[: k_end + 1])
    line(f"      heat still inside the device at the end of the pulse (Cauer): {E_stored_end / 1e3:.2f} kJ of "
         f"{P_PULSE * T_PULSE / 1e3:.1f} kJ")
    mac("cpP", P_PULSE / 1e3, 0)
    mac("cpT", T_PULSE, 1)
    mac("cpE", P_PULSE * T_PULSE / 1e3, 1)
    mac("cpTjI", TjI, 0)
    mac("cpTjC", TjC, 0)
    mac("cpTjF", TjF, 0)
    mac("cpDiff", TjF - TjC, 0)
    mac("cpTcC", max(resC[k_end, 1], resC[k_end, 2]), 0)
    mac("cpTcF", resF_case[k_end], 0)
    mac("cpTcFzero", resF_case[1], 0)
    mac("cpQten", t_edges[k10] * 1e3, 0)
    mac("cpQend", qtot[k_end] / P_PULSE * 100, 0)
    mac("cpEstored", E_stored_end / 1e3, 2)
    mac("cpEstoredPct", E_stored_end / (P_PULSE * T_PULSE) * 100, 0)
    mac("cpRch", R_CH * 1e3, 0)
    mac("cpCbar", C_BAR, 0)
    return out


# ---------------------------------------------------------------- 3. synthetic press-pack
def synthetic_network(sink, layers=None):
    """1D finite-difference press-pack. sink: 'ideal' (both faces at the reference), 'anode' / 'cathode' (that face
    at the reference, the other adiabatic), 'bars' (contact and lumped bar on each face)."""
    layers = LAYERS if layers is None else layers
    A = math.pi / 4 * D_POLE ** 2
    net = fc.Network()
    cells, si = [], []
    pend_R = 0.0                      # resistance waiting to be added to the next link (contact)
    prev, prev_half = None, 0.0
    first = None
    for lay in layers:
        if lay[0] == "contact":
            pend_R += lay[1]
            continue
        mat, th, nc = lay
        k, rc = MAT[mat]
        dx = th / nc
        for i in range(nc):
            nd = net.node(rc * A * dx)
            half = dx / (2 * k * A)
            if prev is None:
                first = (nd, half)
            else:
                net.link(prev, nd, prev_half + pend_R + half)
                pend_R = 0.0
            if mat == "Si":
                si.append(nd)
            prev, prev_half = nd, half
            cells.append(nd)
    last = (prev, prev_half)
    faces = []
    for (nd, half), side in ((first, "anode"), (last, "cathode")):
        if sink == "ideal" or sink == side:
            net.ground(nd, half)
            faces.append(None)
        elif sink == "bars":
            c = net.node(0.0)
            net.link(nd, c, half)
            b = net.node(C_BAR)
            net.link(c, b, R_CH)
            faces.append(c)
        else:                         # adiabatic face
            faces.append(None)
    n = len(net.C)
    w = np.zeros(n)
    w[si] = 1.0 / len(si)
    return net, w, faces


def modal_foster(net, w):
    """Exact Foster expansion of the network for injection and readout with the weights w."""
    m = net.modes(w)
    lam, f = m["lam"], m["f"]
    keep = lam > 0
    Rf = f[keep] ** 2 / lam[keep]
    tf = 1.0 / lam[keep]
    order = np.argsort(tf)[::-1]
    return Rf[order], tf[order]


def fit_foster(t, z, n):
    """Least-squares Foster fit of n terms to the curve z(t) (relative error), positive parameters."""
    def res(x):
        return fc.zth_foster(np.exp(x[:n]), np.exp(x[n:]), t) / z - 1.0
    best = None
    for shift in np.linspace(-0.5, 0.5, 5):
        tau0 = np.logspace(math.log10(t[-1]) - 0.7 + shift, math.log10(t[0]) + 0.3, n)
        R0 = np.full(n, z[-1] / n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = least_squares(res, np.log(np.r_[R0, tau0]), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=40000)
        if best is None or sol.cost < best.cost:
            best = sol
    R, tau = np.exp(best.x[:n]), np.exp(best.x[n:])
    o = np.argsort(tau)[::-1]
    return R[o], tau[o], np.max(np.abs(res(best.x)))


def coupled_rows(Rc, Cc, R_sides, tp):
    """Junction rise per watt at the end of rectangular pulses tp: Cauer per side on the bars."""
    ks, _ = fc.split_sides(Rc, Cc, R_sides)
    net, j, _, _ = coupled_cauer(Rc, Cc, ks)
    return step_response(net, unit(len(net.C), j), tp, [unit(len(net.C), j)])[:, 0]


def foster_on_bars(R, tau, tp):
    netF, cF = bars_common_case()
    return fc.zth_foster(R, tau, tp) + step_response(netF, unit(len(netF.C), cF), tp, [unit(len(netF.C), cF)])[:, 0]


def synthetic(layers, tag, label):
    A = math.pi / 4 * D_POLE ** 2
    netI, w, _ = synthetic_network("ideal", layers)
    Rx, tx = modal_foster(netI, w)
    Rth = Rx.sum()
    R_sides = []
    for side in ("anode", "cathode"):
        net_s, w_s, _ = synthetic_network(side, layers)
        R_sides.append(step_response(net_s, w_s, [1e4], [w_s])[0, 0])
    caps = {}
    for lay in layers:
        if lay[0] != "contact":
            caps.setdefault(lay[0], []).append(MAT[lay[0]][1] * A * lay[1])
    C_si, C_mo, C_cu = sum(caps["Si"]), sum(caps["Mo"]), sum(caps["Cu"])
    line(f"{label}: {len(netI.C)} nodes, R_th(j-c) {Rth * 1e3:.3f} K/kW, single side anode {R_sides[0] * 1e3:.3f} / "
         f"cathode {R_sides[1] * 1e3:.3f} K/kW (parallel {1 / (1 / R_sides[0] + 1 / R_sides[1]) * 1e3:.3f})")
    line(f"   heat capacities: Si {C_si:.2f}, Mo {C_mo:.1f} ({', '.join(f'{c:.1f}' for c in caps['Mo'])}), "
         f"Cu {C_cu:.1f} J/K; total {C_si + C_mo + C_cu:.1f} J/K")
    big = Rx > 1e-12 * Rth
    # exact ladder from the full expansion (bidiagonalization; the polynomial expansion fails here)
    Rcx, Ccx = fc.foster_to_cauer_bidiag(Rx[big], tx[big])
    Rfx, tfx = fc.cauer_to_foster(Rcx, Ccx)
    t_chk = np.logspace(-6, 3, 400)
    err_x = np.max(np.abs(fc.zth_foster(Rfx, tfx, t_chk) / fc.zth_foster(Rx, tx, t_chk) - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Rpx, Cpx = fc.foster_to_cauer(Rx[big], tx[big])
            poly_ok = bool(np.all(Rpx > 0) and np.all(Cpx > 0) and np.all(np.isfinite(Rpx)))
        except Exception:
            poly_ok = False
    cumR, cumC = np.cumsum(Rcx), np.cumsum(Ccx)
    c99 = cumC[np.searchsorted(cumR, 0.99 * Rth) - 1]
    line(f"   exact expansion: {big.sum()} modes with R_i > 1e-12 R_th (tau {tx[big][-1]:.1e} - {tx[0]:.3f} s); exact ladder "
         f"by bidiagonalization, round trip {err_x:.1e}; cumulative C up to 99 % of R_th {c99:.1f} J/K, largest C "
         f"{Ccx.max():.3g} J/K; polynomial expansion {'valid' if poly_ok else 'returns negative or infinite elements'}")
    # datasheet-like curve and fits
    tw = np.logspace(math.log10(T_FIT[0]), math.log10(T_FIT[1]), 81)
    zw = fc.zth_foster(Rx, tx, tw)
    fits = {}
    for nf in N_FITS:
        Rf, tf, err = fit_foster(tw, zw, nf)
        Rc, Cc = fc.foster_to_cauer(Rf, tf)
        Rb, Cb = fc.foster_to_cauer_bidiag(Rf, tf)
        agree = max(np.max(np.abs(Rc / Rb - 1)), np.max(np.abs(Cc / Cb - 1)))
        fits[nf] = dict(R=Rf, tau=tf, Rc=Rc, Cc=Cc, err=err)
        line(f"   fit {nf} terms on {T_FIT[0] * 1e3:.0f} ms - {T_FIT[1]:.0f} s: max error {err * 100:.2f} %; tau "
             f"{np.array2string(tf, precision=4)} s; Cauer C {np.array2string(Cc, precision=2)} J/K "
             f"(sum {Cc.sum():.1f}); polynomial vs bidiagonalization {agree:.0e}")
    f4 = fits[4]
    ks, sides = fc.split_sides(f4["Rc"], f4["Cc"], R_sides)
    # single-side curves: model against the split ladders (4-term fit and exact ladder)
    t = np.logspace(-5, 3, 161)
    ssc = {}
    for side, idx in (("anode", 0), ("cathode", 1)):
        net_s, w_s, _ = synthetic_network(side, layers)
        z_true = step_response(net_s, w_s, t, [w_s])[:, 0]
        zs = {}
        for lname, (Rl, Cl) in (("four", (f4["Rc"], f4["Cc"])), ("exact", (Rcx, Ccx))):
            _, sl = fc.split_sides(Rl, Cl, R_sides)
            net = fc.Network()
            j = net.node(Cl[0])
            for s_i, (Rs, Cs) in enumerate(sl):
                prev = j
                for r, cap in zip(Rs[:-1], Cs):
                    nk = net.node(cap); net.link(prev, nk, r); prev = nk
                if s_i == idx:
                    net.ground(prev, Rs[-1])
            zs[lname] = step_response(net, unit(len(net.C), j), t, [unit(len(net.C), j)])[:, 0]
        ssc[side] = (z_true, zs["four"])
        sel = t >= 1e-3
        for lname in ("four", "exact"):
            e = zs[lname][sel] / z_true[sel] - 1
            k = int(np.argmax(np.abs(e)))
            line(f"   single side {side}, {'4-term' if lname == 'four' else 'exact'} ladder split: model "
                 f"{z_true[-1] * 1e3:.2f} K/kW, split {zs[lname][-1] * 1e3:.2f} K/kW; extreme deviation from 1 ms on "
                 f"{e[k] * 100:+.1f} % at {t[sel][k]:.2f} s")
            mac(f"{tag}S{side[0]}Dev{'F' if lname == 'four' else 'X'}", e[k] * 100, 0 if abs(e[k]) >= 0.02 else 1)
            mac(f"{tag}S{side[0]}Abs{'F' if lname == 'four' else 'X'}", abs(e[k]) * 100, 0 if abs(e[k]) >= 0.02 else 1)
            if lname == "four":
                mac(f"{tag}S{side[0]}DevT", t[sel][k], 1)
    # junction on the bars
    tp = np.logspace(-3, 2, 101)
    netB, wB, _ = synthetic_network("bars", layers)
    z_model = step_response(netB, wB, tp, [wB])[:, 0]
    zC = {nf: coupled_rows(fd["Rc"], fd["Cc"], R_sides, tp) for nf, fd in fits.items()}
    zCx = coupled_rows(Rcx, Ccx, R_sides, tp)
    zF = foster_on_bars(f4["R"], f4["tau"], tp)
    zI = fc.zth_foster(Rx, tx, tp)
    line(f"   junction on the bars (contact {R_CH * 1e3:.0f} K/kW, {C_BAR:.0f} J/K per side), rise per kW at the end "
         f"of a rectangular pulse:")
    for tq in (0.01, 0.1, 0.5, 1.0, 10.0):
        k = np.argmin(np.abs(tp - tq))
        line(f"      {tp[k]:7.3f} s: model {z_model[k] * 1e3:6.3f}, Cauer 4 terms per side {zC[4][k] * 1e3:6.3f} "
             f"({(zC[4][k] / z_model[k] - 1) * 100:+.1f} %), exact ladder per side {zCx[k] * 1e3:6.3f} "
             f"({(zCx[k] / z_model[k] - 1) * 100:+.1f} %), Foster {zF[k] * 1e3:6.3f} "
             f"({(zF[k] / z_model[k] - 1) * 100:+.1f} %), ideal sink {zI[k] * 1e3:6.3f} K/kW")
    dev = lambda z: np.max(np.abs(z / z_model - 1)) * 100
    for nf in N_FITS:
        line(f"      fit {nf} terms, Cauer per side: largest deviation from the model {dev(zC[nf]):.1f} %")
    line(f"      exact ladder per side: largest deviation {dev(zCx):.1f} %; Foster: largest {dev(zF):.0f} % "
         f"(at {tp[np.argmax(zF / z_model)]:.3f} s)")
    # short times: the fitted ladder against the model below the window
    shorts = {}
    for tq in (1e-4, 1e-5):
        zt = fc.zth_foster(Rx, tx, tq)
        z4 = fc.zth_foster(f4["R"], f4["tau"], tq)
        shorts[tq] = (z4 / zt - 1) * 100
        line(f"   below the window, {tq * 1e6:.0f} us: model {zt * 1e3:.4f} K/kW, 4-term fit {z4 * 1e3:.4f} K/kW "
             f"({shorts[tq]:+.0f} %)")
    kk = np.argmin(np.abs(tp - 0.5))
    mac(f"{tag}Nodes", len(netI.C))
    mac(f"{tag}Rth", Rth * 1e3, 2)
    mac(f"{tag}Ra", R_sides[0] * 1e3, 1)
    mac(f"{tag}Rc", R_sides[1] * 1e3, 1)
    mac(f"{tag}Csi", C_si, 2)
    mac(f"{tag}Cmo", C_mo, 1)
    mac(f"{tag}Ccu", C_cu, 0)
    mac(f"{tag}Ctot", C_si + C_mo + C_cu, 0)
    mac(f"{tag}Modes", int(big.sum()))
    mac(f"{tag}CninetyNine", c99, 0)
    mac(f"{tag}Cmax", sci_tex(Ccx.max(), 0), None)
    mac(f"{tag}CoFull", Ccx[0], 2)
    mac(f"{tag}ErrFull", sci_tex(max(err_x, 1e-17)), None)
    for nf in N_FITS:
        fd = fits[nf]
        tn = {3: "Three", 4: "Four", 5: "Five", 6: "Six"}[nf]
        mac(f"{tag}Fit{tn}Err", fd["err"] * 100, 2)
        mac(f"{tag}Fit{tn}Co", fd["Cc"][0], 2)
        mac(f"{tag}Fit{tn}Csum", fd["Cc"].sum(), 0)
        mac(f"{tag}Fit{tn}Tmin", fd["tau"][-1] * 1e3, 2)
        mac(f"{tag}Fit{tn}Dev", dev(zC[nf]), 1)
    for i in range(4):
        mac(f"{tag}FR{'ABCD'[i]}", f4["R"][i] * 1e3, 3)
        mac(f"{tag}FT{'ABCD'[i]}", f4["tau"][i], 4)
        mac(f"{tag}RcF{'ABCD'[i]}", f4["Rc"][i] * 1e3, 3)
        mac(f"{tag}CcF{'ABCD'[i]}", f4["Cc"][i], 2 if f4["Cc"][i] < 10 else 1)
    mac(f"{tag}Ka", ks[0], 3)
    mac(f"{tag}Kc", ks[1], 3)
    for key, z in (("Zm", z_model), ("Zc", zC[4]), ("Zx", zCx), ("Zf", zF), ("Zi", zI)):
        mac(f"{tag}Half{key}", z[kk] * 1e3, 2)
    mac(f"{tag}HalfEc", (zC[4][kk] / z_model[kk] - 1) * 100, 1)
    mac(f"{tag}HalfEx", (zCx[kk] / z_model[kk] - 1) * 100, 1)
    mac(f"{tag}HalfEf", (zF[kk] / z_model[kk] - 1) * 100, 0)
    mac(f"{tag}HalfEi", (zI[kk] / z_model[kk] - 1) * 100, 1)
    mac(f"{tag}MaxEc", dev(zC[4]), 1)
    mac(f"{tag}MinEc", np.min(zC[4] / z_model - 1) * 100, 2)
    mac(f"{tag}MaxEx", dev(zCx), 1 if dev(zCx) < 10 else 0)
    mac(f"{tag}MaxEf", dev(zF), 0)
    mac(f"{tag}TmaxEf", tp[np.argmax(zF / z_model)] * 1e3, 0)
    mac(f"{tag}ShortA", shorts[1e-4], 0)
    mac(f"{tag}ShortB", shorts[1e-5], 0)
    return dict(Rx=Rx, tx=tx, Rcx=Rcx, Ccx=Ccx, fits=fits, ssc=ssc, t=t, tp=tp, z_model=z_model, zC=zC, zCx=zCx,
                zF=zF, zI=zI, tw=tw, zw=zw, caps=dict(Si=C_si, Mo=C_mo, Cu=C_cu), R_sides=R_sides)


# ---------------------------------------------------------------- 4. accuracy of the two algorithms
def accuracy_study():
    res = {}
    for dec in STUDY_DEC:
        ep, eb = [], []
        for n in STUDY_N:
            tau = np.logspace(0, -dec, n)
            R = np.full(n, 1.0 / n)
            Rm, Cm = fc.foster_to_cauer_mp(R, tau, MP_DIGITS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    Rp, Cp = fc.foster_to_cauer(R, tau)
                    e = max(np.max(np.abs(Rp / Rm - 1)), np.max(np.abs(Cp / Cm - 1)))
                    bad = not (np.all(Rp > 0) and np.all(Cp > 0) and np.isfinite(e))
                except Exception:
                    e, bad = np.inf, True
            ep.append((e, bad))
            Rb, Cb = fc.foster_to_cauer_bidiag(R, tau)
            eb.append(max(np.max(np.abs(Rb / Rm - 1)), np.max(np.abs(Cb / Cm - 1))))
        res[dec] = (ep, eb)
        first_bad = next((n for n, (e, b) in zip(STUDY_N, ep) if b), None)
        first_1pct = next((n for n, (e, b) in zip(STUDY_N, ep) if e > 1e-2), None)
        line(f"   {dec} decade(s): polynomial expansion above 1e-6 from n = "
             f"{next((n for n, (e, b) in zip(STUDY_N, ep) if e > 1e-6), '-')}, above 1 % from n = {first_1pct or '-'}, "
             f"negative or infinite elements from n = {first_bad or '-'}; bidiagonalization at most {max(eb):.1e}")
        tagd = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}[dec]
        mac(f"acBad{tagd}", first_bad if first_bad else "--")
        mac(f"acPct{tagd}", first_1pct if first_1pct else "--")
        mac(f"acSix{tagd}", next((n for n, (e, b) in zip(STUDY_N, ep) if e > 1e-6), "--"))
        mac(f"acBid{tagd}", sci_tex(max(eb), 0), None)
    allb = max(max(v[1]) for v in res.values())
    mac("acBidMax", sci_tex(allb, 0), None)
    return res


# ---------------------------------------------------------------- figures
# Each figure is drawn twice: English labels for foster_to_cauer.tex (figures/fig_*.pdf) and Italian labels for
# foster_to_cauer_it.tex (figures/fig_*_it.pdf). L(en, it) returns the label in the current language.
LANG = "en"


def L(en, it):
    return it if LANG == "it" else en


def loc_num(s):
    """A number written in a label: decimal point in English, decimal comma in Italian."""
    return s.replace(".", ",") if LANG == "it" else s


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + ("_it" if LANG == "it" else "") + ".pdf"))
    plt.close(fig)


def fig_datasheet(dd, dt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw=dict(wspace=0.32))
    t = dd["t"]
    sel = (t >= 1e-4) & (t <= 1e2)
    for d, col, lab in ((dd, C_A, DIODE["name"]), (dt, C_B, THYR["name"])):
        a1.loglog(t[sel], d["zf"][sel] * 1e3, color=col, lw=1.4, label=f"{lab}, Foster (datasheet)")
        tm = np.logspace(-4, 2, 25)
        a1.loglog(tm, fc.zth_cauer(d["Rc"], d["Cc"], tm) * 1e3, "o", ms=3.2, mfc="white", mec=col, mew=0.9,
                  label=L(f"{lab}, Cauer ladder", f"{lab}, rete a scala di Cauer"))
    tt = np.logspace(-4, -2.4, 10)
    a1.loglog(tt, tt / dd["Cc"][0] * 1e3, ls=(0, (3, 2)), color=INK2, lw=0.9)
    a1.text(1.3e-4, 2.0e-2, r"$t/C_1$", color=INK2, fontsize=7.5)
    a1.set_xlabel(L("time (s)", "tempo (s)"))
    a1.set_ylabel(r"$Z_\mathrm{th(j-c)}$ (K/kW)")
    a1.set_xlim(1e-4, 1e2)
    a1.set_ylim(1e-2, 2e1)
    a1.legend(loc="lower right")
    a1.set_title(L("(a) step response, double-side cooled", "(a) risposta al gradino, raffreddamento su due lati"),
                 loc="left")
    for d, col, lab in ((dd, C_A, DIODE["name"]), (dt, C_B, THYR["name"])):
        x, y = fc.structure_function(d["Rc"] * 1e3, d["Cc"])
        a2.semilogy(x, y, color=col, lw=1.4, label=lab)
        Rs = np.concatenate([[0.0], np.cumsum(d["Rc"] * 1e3)])
        Cs = np.cumsum(d["Cc"])
        above = col == C_A
        for k in range(4):
            txt = loc_num(f"{Cs[k]:.0f}" if Cs[k] >= 10 else f"{Cs[k]:.1f}")
            if above:
                a2.text(0.5 * (Rs[k] + Rs[k + 1]), Cs[k] * 1.15, txt, color=col, fontsize=6.8, ha="center", va="bottom")
            elif Rs[k + 1] - Rs[k] < 0.6 and k > 0:       # narrow step: label above it
                a2.text(0.5 * (Rs[k] + Rs[k + 1]), Cs[k] * 1.15, txt, color=col, fontsize=6.8, ha="center", va="bottom")
            else:
                a2.text(0.5 * (Rs[k] + Rs[k + 1]), Cs[k] / 1.15, txt, color=col, fontsize=6.8, ha="center", va="top")
    a2.set_xlabel(L(r"cumulative resistance from the junction, $\sum R_k$ (K/kW)",
                    r"resistenza cumulata dalla giunzione, $\sum R_k$ (K/kW)"))
    a2.set_ylabel(L(r"cumulative capacitance $\sum C_k$ (J/K)", r"capacità cumulata $\sum C_k$ (J/K)"))
    a2.set_xlim(0, 8.5)
    a2.set_ylim(1, 500)
    a2.legend(loc="lower right")
    a2.set_title(L("(b) Cauer ladders as structure functions", "(b) reti a scala di Cauer come funzioni di struttura"),
                 loc="left")
    save(fig, "fig_datasheet")


def fig_coupling(cp):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw=dict(wspace=0.3))
    t = cp["t"] * 1e3
    a1.plot(t, cp["TjF"], color=C_B, label=L("junction, Foster on the bars", "giunzione, Foster sulle barre"))
    a1.plot(t, cp["TjC"], color=C_A, label=L("junction, Cauer per side on the bars", "giunzione, Cauer per lato sulle barre"))
    a1.plot(t, cp["TjI"], color=INK2, ls=(0, (4, 2)), lw=1.0,
            label=L("junction, ideal heat sink", "giunzione, dissipatore ideale"))
    a1.plot(t, cp["TcF"], color=C_B, lw=0.9, ls=(0, (1, 1.5)), label="case, Foster")
    a1.plot(t, cp["TcC"].max(1), color=C_A, lw=0.9, ls=(0, (1, 1.5)),
            label=L("case (warmer side), Cauer", "case (lato più caldo), Cauer"))
    a1.set_xlim(0, 2000)
    a1.set_ylim(0, None)
    a1.set_xlabel(L("time (ms)", "tempo (ms)"))
    a1.set_ylabel(L("temperature rise (K)", "sovratemperatura (K)"))
    a1.legend(loc="upper right", fontsize=6.6)
    a1.set_title(f"(a) {P_PULSE / 1e3:.0f} kW {L('for', 'per')} {T_PULSE * 1e3:.0f} ms, {DIODE['name']}", loc="left")
    q = cp["qC"] / P_PULSE * 100
    m = cp["t"] <= T_PULSE + 1e-9
    a2.plot(t[m], np.full(m.sum(), 100.0), color=C_B,
            label=L("Foster network (both sides together)", "rete di Foster (i due lati insieme)"))
    a2.plot(t[m], q[m].sum(1), color=C_A, label=L("Cauer ladders, both sides", "reti a scala di Cauer, due lati"))
    a2.plot(t[m], q[m, 0], color=C_A, lw=0.9, ls=(0, (3, 1.5)), label=L("Cauer, anode side", "Cauer, lato anodo"))
    a2.plot(t[m], q[m, 1], color=C_A, lw=0.9, ls=(0, (1, 1.5)), label=L("Cauer, cathode side", "Cauer, lato catodo"))
    a2.set_xlim(0, T_PULSE * 1e3)
    a2.set_ylim(0, 110)
    a2.set_xlabel(L("time (ms)", "tempo (ms)"))
    a2.set_ylabel(L("heat into the bars (% of the power)", "calore nelle barre (% della potenza)"))
    a2.legend(loc="center right", fontsize=6.6)
    a2.set_title(L("(b) heat leaving the device during the pulse", "(b) calore che esce dal dispositivo nell'impulso"),
                 loc="left")
    save(fig, "fig_coupling")


def fig_synthetic(sy, ss):
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.4), gridspec_kw=dict(wspace=0.3, hspace=0.42))
    (a1, a2), (a3, a4) = axs
    t = sy["t"]
    zx = fc.zth_foster(sy["Rx"], sy["tx"], t)
    a1.axvspan(T_FIT[0], T_FIT[1], color=GRID, alpha=0.5, lw=0)
    a1.text(1.15e-3, 2.0e1, L("window of the fit", "finestra del fit"), color=INK2, fontsize=6.5)
    a1.loglog(t, zx * 1e3, color=INK, lw=1.5, label=L("model, both sides cooled", "modello, due lati raffreddati"))
    f4 = sy["fits"][4]
    a1.loglog(t, fc.zth_foster(f4["R"], f4["tau"], t) * 1e3, color=C_A, ls=(0, (4, 2)), lw=1.1,
              label=L("4-term Foster fit (= its Cauer ladder)", "fit a 4 termini (= sua rete di Cauer)"))
    for side, side_it, col in (("anode", "anodo", C_C), ("cathode", "catodo", C_D)):
        zt, zs = sy["ssc"][side]
        a1.loglog(t, zt * 1e3, color=col, lw=1.3,
                  label=L(f"model, {side} side cooled only", f"modello, raffreddato solo lato {side_it}"))
        a1.loglog(t[::8], zs[::8] * 1e3, "o", ms=2.8, mfc="white", mec=col, mew=0.8,
                  label=L(f"split ladders, {side} side only", f"reti a scala suddivise, solo lato {side_it}"))
    a1.set_xlim(1e-5, 1e2)
    a1.set_ylim(3e-3, 4e1)
    a1.set_xlabel(L("time (s)", "tempo (s)"))
    a1.set_ylabel(r"$Z_\mathrm{th(j-c)}$ (K/kW)")
    a1.legend(loc="lower right", fontsize=5.9)
    a1.set_title(L("(a) step responses, sides with different layers", "(a) risposte al gradino, lati con strati diversi"),
                 loc="left")
    x, y = fc.structure_function(sy["Rcx"] * 1e3, sy["Ccx"])
    a2.semilogy(x, y, color=INK, lw=1.4, label=L(f"exact ladder ({len(sy['Rcx'])} sections)",
                                                 f"rete a scala esatta ({len(sy['Rcx'])} sezioni)"))
    for nf, col in zip((3, 4, 6), (RAMP[0], C_A, RAMP[3])):
        fd = sy["fits"][nf]
        x, y = fc.structure_function(fd["Rc"] * 1e3, fd["Cc"])
        a2.semilogy(x, y, color=col, lw=1.1, label=L(f"ladder of the {nf}-term fit", f"rete a scala del fit a {nf} termini"))
    c = sy["caps"]
    for val, lab, xt in ((c["Si"], "Si", 6.3), (c["Si"] + c["Mo"], "Si + Mo", 6.3),
                         (c["Si"] + c["Mo"] + c["Cu"], L("all layers", "tutti gli strati"), 0.12)):
        a2.axhline(val, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        a2.text(xt, val * 1.12, lab, color=INK2, fontsize=6.5)
    a2.set_xlim(0, 8.2)
    a2.set_ylim(0.04, 800)
    a2.set_xlabel(L(r"cumulative resistance from the junction, $\sum R_k$ (K/kW)",
                    r"resistenza cumulata dalla giunzione, $\sum R_k$ (K/kW)"))
    a2.set_ylabel(L(r"cumulative capacitance $\sum C_k$ (J/K)", r"capacità cumulata $\sum C_k$ (J/K)"))
    a2.legend(loc="lower right", fontsize=6.0)
    a2.set_title(L("(b) structure functions, sides with different layers", "(b) funzioni di struttura, lati con strati diversi"),
                 loc="left")
    for a, d, title in ((a3, sy, L("(c) on the bars, sides with different layers", "(c) sulle barre, lati con strati diversi")),
                        (a4, ss, L("(d) on the bars, sides with equal layers", "(d) sulle barre, lati con strati uguali"))):
        tp = d["tp"]
        a.loglog(tp, d["zF"] * 1e3, color=C_B, lw=1.2, label=L("Foster (4-term fit) on the bars", "Foster (fit a 4 termini) sulle barre"))
        a.loglog(tp, d["z_model"] * 1e3, color=INK, lw=2.2, alpha=0.35, label=L("model on the bars", "modello sulle barre"))
        a.loglog(tp, d["zC"][4] * 1e3, color=C_A, lw=1.2, ls=(0, (4, 2)),
                 label=L("Cauer per side (4-term fit)", "Cauer per lato (fit a 4 termini)"))
        a.loglog(tp, d["zCx"] * 1e3, color=C_C, lw=1.2, ls=(0, (1, 1.2)),
                 label=L("exact ladder, per side", "rete a scala esatta, per lato"))
        a.loglog(tp, d["zI"] * 1e3, color=INK2, lw=0.9, ls=(0, (5, 2, 1, 2)), label=L("ideal heat sink", "dissipatore ideale"))
        if d is sy:
            a.annotate(L("exact ladder: no better\nthan an ideal heat sink", "rete a scala esatta:\nnon meglio di un\ndissipatore ideale"),
                       xy=(20, d["zCx"][-21] * 1e3), xytext=(2.0, 1.2),
                       fontsize=6.3, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        a.set_xlim(1e-3, 1e2)
        a.set_ylim(0.2, 200)
        a.set_xlabel(L("length of the rectangular power pulse (s)", "durata dell'impulso rettangolare di potenza (s)"))
        a.set_ylabel(L("junction rise at the end (K/kW)", r"$\Delta T_\mathrm{j}$ a fine impulso (K/kW)"))
        a.legend(loc="upper left", fontsize=6.0)
        a.set_title(title, loc="left")
    save(fig, "fig_synthetic")


def fig_accuracy(ac):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw=dict(wspace=0.25), sharey=True)
    for dec, col in zip(STUDY_DEC, RAMP):
        ep, eb = ac[dec]
        e = np.array([min(x, 1e3) if np.isfinite(x) else 1e3 for x, _ in ep])
        bad = np.array([b for _, b in ep])
        lab = L(f"{dec} decade{'s' if dec > 1 else ''}", f"{dec} decad{'i' if dec > 1 else 'e'}")
        a1.semilogy(STUDY_N, np.maximum(e, 1e-17), color=col, lw=1.2, label=lab)
        if bad.any():
            a1.semilogy(np.array(STUDY_N)[bad], np.maximum(e[bad], 1e-17), "x", color=col, ms=3.5, mew=0.9)
        a2.semilogy(STUDY_N, np.maximum(eb, 1e-17), color=col, lw=1.2, label=lab)
    for a, title in ((a1, L("(a) polynomial expansion (foster_to_cauer)", "(a) sviluppo polinomiale (foster_to_cauer)")),
                     (a2, L("(b) bidiagonalization", "(b) bidiagonalizzazione"))):
        a.set_xlabel(L("number of Foster terms", "numero di termini di Foster"))
        a.set_xlim(0, 60)
        a.set_ylim(1e-17, 1e4)
        a.axhline(1e-2, color=MUTED, lw=0.7, ls=(0, (2, 2)))
        a.set_title(title, loc="left")
    a1.set_ylabel(L("largest relative error of the elements", "massimo errore relativo degli elementi"))
    a1.text(1, 1.6e-2, "1 %", color=INK2, fontsize=6.5)
    a1.text(59, 3e-17, L("x: negative or infinite elements", "x: elementi negativi o infiniti"), color=INK2, fontsize=6.5,
            ha="right", va="bottom")
    a2.legend(loc="upper left", title=L("time constants spread over", "costanti di tempo distribuite su"), title_fontsize=6.8)
    save(fig, "fig_accuracy")


# ---------------------------------------------------------------- main
def main():
    global LANG
    t0 = time.time()
    line("Foster and Cauer thermal networks: worked examples")
    line()
    line("1. Datasheet networks, junction to case, double-side cooled")
    dd = device(DIODE, "dd")
    dt = device(THYR, "dt")
    line()
    line(f"2. {DIODE['name']} on two copper bars")
    cp = coupling_example(dd)
    line()
    line("3. Synthetic press-pack, pole of %.0f mm" % (D_POLE * 1e3))
    sy = synthetic(LAYERS, "sy", "sides with different layers")
    ss = synthetic(LAYERS_SYM, "ss", "sides with equal layers")
    line()
    line(f"4. Accuracy against {MP_DIGITS} digits, equal R, time constants log-spaced over 1 to 8 decades, n = "
         f"{STUDY_N[0]} to {STUDY_N[-1]}")
    ac = accuracy_study()
    mac("mpDigits", MP_DIGITS)
    for LANG in ("en", "it"):              # figures/fig_*.pdf, then figures/fig_*_it.pdf
        fig_datasheet(dd, dt)
        fig_coupling(cp)
        fig_synthetic(sy, ss)
        fig_accuracy(ac)
    line()
    line(f"done in {time.time() - t0:.1f} s")
    with open(os.path.join(HERE, "foster_cauer_results.txt"), "w") as fo:
        fo.write("\n".join(LOG) + "\n")
    with open(os.path.join(HERE, "foster_cauer_values.tex"), "w") as fo:
        fo.write("% generated by foster_cauer_calc.py: do not edit by hand\n")
        for k, v in MAC.items():
            fo.write(f"\\newcommand{{\\fc{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()

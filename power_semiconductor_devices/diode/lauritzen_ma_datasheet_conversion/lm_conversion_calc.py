"""Worked examples of lauritzen_ma_datasheet_conversion.tex: from the datasheet of a diode to the parameters of
the Lauritzen-Ma model and back.

The script applies the relations of the note (module lm_conversion.py) to two assumed diodes and writes every
number quoted in the note:
  1. general relations: factors of the definitions of the reverse recovery time, accuracy of the long-ramp
     closed form, what the closed form returns when the ramp is short;
  2. diode A, a rectifier diode tested on a long ramp, and diode B, a fast diode tested on a short ramp:
     lifetime and transit time from the peak reverse current and the recovered charge, parameters of the
     forward characteristic, leakage resistance, equivalent data for other parameter sets, sensitivities;
  3. check of each set in a simulated commutation cell (model with junction capacitance and RC snubber);
  4. the model away from the datasheet point: slope, forward current, temperature, resistive test.

The two diodes are not products: their data are round values of the size found in datasheets of the two classes.
To convert a real diode, replace the parameters below with the data of its datasheet.

Outputs, next to the script: lm_conversion_results.txt (log), lm_conversion_values.tex (macros \\lm... of the
note, read by both language versions), figures/fig_*.pdf and figures/fig_*_it.pdf (the same figures with
Italian labels, for lauritzen_ma_datasheet_conversion_it.tex). Only numpy, scipy and matplotlib are needed (numpy 1.x and 2.x). A few seconds.
Run:  python lm_conversion_calc.py"""
import math, os, re, time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import brentq
import lm_conversion as lm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- parameters of the worked examples
# Diode A: assumed datasheet of a 4.5 kV press-pack rectifier diode (line-frequency rectifiers)
A = dict(
    VRRM=4500.0, IRRM=50e-3,            # V, A: repetitive peak reverse voltage, reverse current at VRRM and T
    T=150.0,                            # C: junction temperature of all the data below
    VT0=0.85, RT=0.60e-3,               # V, Ohm: threshold voltage and slope resistance of the straight line
    I1=500.0, I2=1500.0,                # A: currents at which the straight line crosses the characteristic
    N=2.0,                              # emission coefficient, assumed (the straight line does not give it)
    IF=1000.0, A=5e6, VR=100.0,         # A, A/s, V: test conditions of the recovery data
    IRR=125.0, QRR=3200e-6,             # A, C: peak reverse recovery current and recovered charge (maximum)
    Q_EXP=0.30, Q_RANGE=(0.5e6, 50e6),  # curve of the charge against the slope: exponent and range in A/s
    CJ0=20e-9,                          # F: zero-bias junction capacitance (estimate, not a datasheet value)
    RSN=10.0, CSN=0.47e-6,              # Ohm, F: snubber of the simulated test cell
    T2=100.0,                           # C: second temperature; tau and TM scaled with the absolute temperature
                                        #    (an assumption of the example, to show the effect of the scaling)
    D=12e-4)                            # m2/s: ambipolar diffusion constant at T, for the plausibility check of TM
# Diode B: assumed datasheet of a 1200 V fast soft-recovery diode (freewheeling diode of a switched leg)
B = dict(
    VRRM=1200.0, IRRM=1e-3, T=125.0,
    VF=((25.0, 1.18), (50.0, 1.40), (100.0, 1.80)),   # (A, V): three points of the typical forward curve
    IF=100.0, A=1e9, VR=600.0,
    IRR=80.0, QRR=12e-6,                # typical values
    CJ0=1e-9,
    RSN=5.6, CSN=22e-9,
    T2=25.0, IRR2=55.0, QRR2=6e-6,      # second temperature with its own recovery data (same IF and slope)
    STEP=(0.5, 1.0, 0.25),              # A: resistive test IF, IR, current at which the time is read
    D=13e-4)
P_END = 0.02                            # the recovered charge is integrated down to this fraction of Irr
X_LONG = 3.0                            # (IF + Irr)/(a*tau) above which the ramp is called long in the note

# ---------------------------------------------------------------- style of the figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_NOM, C_WEAK, C_3, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"       # categorical colours, fixed order
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal ramp, light to dark
BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none")
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

LOG, MAC = [], {}


def line(s=""):
    print(s)
    LOG.append(s)


def rnd(value, d=1):
    """number with d decimals for the note; a half is rounded away from zero, as a reader does by hand."""
    return str(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def put(name, value, d=1):
    """macro \\lm<name> of the note (letters only)."""
    assert name.isalpha(), name
    MAC[name] = value if isinstance(value, str) else rnd(value, d)


def sci(x, d=1):
    """number in the form m x 10^e for the note."""
    e = int(math.floor(math.log10(abs(x))))
    return f"{x/10**e:.{d}f}\\times10^{{{e}}}"


def pct(x, d=1):
    """signed percentage for the note, with a proper minus sign."""
    return ("+" if x >= 0 else "-") + rnd(abs(x)*100, d)


# ---------------------------------------------------------------- 1. general relations
def x_of(r, s):
    """x = T1/tau of the diode that gives IF/Irr = r with the softness s = 2*tau_rr/ta (exact relation)."""
    return brentq(lambda x: (1 + r)/x - s/2 - 1/(1 - math.exp(-x)), 1e-9, 1e4, xtol=1e-14)


def closed_form_back(r, s):
    """Data with IF/Irr = r and softness s converted with the long-ramp closed form; returns what that set gives
    back at the same slope and forward current, relative to the data: (Irr'/Irr, Qrr'/Qrr, tau_LR/tau, TM_LR/TM)."""
    ta, trr = 1.0, s/2                                       # units: ta = 1, a = 1, so Irr = 1 and IF = r
    tauL = trr + ta
    TML = tauL*trr/ta
    tau, TM, _ = lm.lm_from_tail(1.0, r, 1.0, trr)
    back = lm.ramp_recovery(1.0, r, tauL, TML)
    return back["Irr"], back["Qrr"]/(0.5 + trr), tauL/tau, TML/TM


def section_general():
    line("1. General relations")
    k_line, k10, k25 = lm.trr_factor("line"), lm.trr_factor("point", p=0.1), lm.trr_factor("point", p=0.25)
    line(f"   fall time of an exponential tail, t_b = kappa*tau_rr: line through 90 % and 25 %: kappa = {k_line:.4f}; "
         f"instant at 10 %: {k10:.4f}; instant at 25 %: {k25:.4f}; triangle with the same charge: 2")
    put("KLine", k_line, 3); put("KTen", k10, 3); put("KQuarter", k25, 3); put("KLineHalf", k_line/2, 2)
    put("KLineLossPct", (1 - k_line/2)*100, 0); put("KTenGainPct", (k10/2 - 1)*100, 0)
    put("KSpread", k10 - k25, 2)
    put("TEndTaus", math.log(1/P_END), 1); put("PEndPct", P_END*100, 0); put("TailFrac", 1 - P_END, 2)
    put("SofStretchThree", 2*(3 - 1)/k10, 2)
    line(f"   stretch factor 3 of the recovery time read at 10 %: softness s = {2*(3 - 1)/k10:.3f}")
    # accuracy of the long-ramp closed form against the length of the ramp x = T1/tau
    line("   long-ramp factor F = Irr/(a*(tau - tau_rr)) = 1 - exp(-x), and Qrr relative to the long-ramp value:")
    for x in (2.0, 3.0, 4.0, 5.0):
        F = 1 - math.exp(-x)
        q = [F*(F*(1 - rho) + 2*rho)/(1 + rho) for rho in (0.1, 0.5)]
        line(f"     x = {x:.0f}: F = {F:.4f} ({(F - 1)*100:+.1f} %), Qrr {(q[0] - 1)*100:+.1f} % (rho = 0.1) to "
             f"{(q[1] - 1)*100:+.1f} % (rho = 0.5)")
        tag = {2.0: "Two", 3.0: "Three", 4.0: "Four", 5.0: "Five"}[x]
        put("FDef" + tag, (1 - F)*100, 1); put("QDefLo" + tag, (1 - q[1])*100, 1); put("QDefHi" + tag, (1 - q[0])*100, 1)
    put("XLong", X_LONG, 0)
    # charge recovered at the terminals as a fraction of the charge stored at the current zero (long ramp)
    for rho, tag in ((0.2, "Lo"), (0.35, "Mid"), (0.5, "Hi")):
        put("QfracRho" + tag, (1 - rho**2)/2*100, 0); put("SofRho" + tag, 2*rho/(1 - rho), 2)
    # what the closed form returns when the ramp is not long, in terms of the data only
    line("   data converted with the long-ramp closed form: what the set returns at the datasheet point")
    line("     Irr/IF     s    x=T1/tau   Irr'/Irr   Qrr'/Qrr   tau_LR/tau   TM_LR/TM")
    for s in (0.5, 1.0, 2.0, 4.0):
        for q in (0.1, 0.2, 0.3, 0.5, 1.0):
            iI, qQ, tt, tm = closed_form_back(1/q, s)
            line(f"     {q:5.2f}  {s:4.1f}   {x_of(1/q, s):7.3f}   {iI:8.4f}   {qQ:8.4f}   {tt:9.4f}   {tm:8.4f}")
    # the same at Irr/IF = 0.1: largest error of the peak and of the charge, for a softness up to 2 and for 4
    worst = lambda s: max(1 - v for v in closed_form_back(10.0, s)[:2])*100
    up = lambda v: math.ceil(v*10 - 1e-9)/10                 # a bound is rounded up
    put("TenthErrTwo", up(max(worst(s) for s in np.linspace(0.25, 2.0, 36))), 1); put("TenthErrFour", up(worst(4.0)), 1)
    line(f"   at Irr/IF = 0.1 the closed-form set returns peak and charge within {MAC['TenthErrTwo']} % for s <= 2 "
         f"and within {MAC['TenthErrFour']} % for s = 4")
    lim = {}
    for err, tag in ((0.01, "One"), (0.05, "Five")):
        vals = []
        for s in (0.5, 1.0, 2.0, 4.0):
            q = brentq(lambda q: closed_form_back(1/q, s)[0] - (1 - err), 1e-3, 5.0, xtol=1e-12)
            vals.append(q)
        lim[tag] = vals
        line(f"   Irr returned within {err*100:.0f} % of the data for Irr/IF below "
             + ", ".join(f"{v:.3f} (s = {s:g})" for v, s in zip(vals, (0.5, 1.0, 2.0, 4.0))))
        put("RatioLo" + tag, min(vals), 2); put("RatioHi" + tag, max(vals), 2)
        for v, st in zip(vals, ("Half", "One", "Two", "Four")):
            put("Ratio" + tag + st, v, 2)
    # sensitivity of the closed form to the data (long ramp), for three values of the softness
    for s, tag in ((0.5, "Half"), (1.0, "One"), (2.0, "Two")):
        put("SensTrrQ" + tag, (1 + s)/s, 1); put("SensTrrI" + tag, (2 + s)/s, 1); put("SensTrrA" + tag, 1/s, 1)
        put("SensTauQ" + tag, (1 + s)/(2 + s), 2); put("SensTauI" + tag, s/(2 + s), 2); put("SensTauA" + tag, 1/(2 + s), 2)
        put("SensTmQ" + tag, (1 + s)/(2 + s) + (1 + s)/s, 1); put("SensTmI" + tag, s/(2 + s) + (2 + s)/s + 1, 1)
        put("SensTmA" + tag, 1/s + 1 - 1/(2 + s), 1)
    line()
    return lim


# ---------------------------------------------------------------- 2. conversion of one diode
def convert(d, key, us):
    """Conversion of the recovery data of one diode; us = 1e6 for times in microseconds, 1e9 for nanoseconds."""
    a, IF, Irr, Qrr = d["A"], d["IF"], d["IRR"], d["QRR"]
    unit = "us" if us == 1e6 else "ns"
    VT = lm.thermal_voltage(d["T"])
    ta = Irr/a
    trr = Qrr/Irr - ta/2
    tau, TM, it = lm.lm_from_tail(a, IF, Irr, trr)
    tauL, TML, _ = lm.lm_from_tail(a, IF, Irr, trr, long_ramp=True)
    r, rL = lm.ramp_recovery(a, IF, tau, TM), lm.ramp_recovery(a, IF, tauL, TML)
    In, Qn = lm.integrate_model(a, IF, tau, TM)
    tau98, TM98, trr98 = lm.lm_from_irr_qrr(a, IF, Irr, Qrr, tail_fraction=1 - P_END)
    mI, mQ = lm.ramp_exponents(a, IF, tau, TM)
    line(f"   recovery data: IF = {IF:g} A, a = {a/1e6:g} A/us, Irr = {Irr:g} A, Qrr = {Qrr*1e6:g} uC, VR = {d['VR']:g} V, "
         f"T = {d['T']:g} C")
    line(f"   ta = Irr/a = {ta*us:.2f} {unit}; triangle charge Irr*ta/2 = {0.5*Irr*ta*1e6:.2f} uC "
         f"({0.5*Irr*ta/Qrr*100:.0f} % of Qrr); tail time constant tau_rr = {trr*us:.2f} {unit}; softness s = {r['s']:.3f}")
    line(f"   exact conversion ({it} iterations): tau = {tau*us:.2f} {unit}, TM = {TM*us:.2f} {unit}, rho = {r['rho']:.3f}, "
         f"IF/(a*tau) = {r['lam']:.3f}, T1/tau = {r['x']:.3f}, F = {r['F']:.4f}")
    line(f"   long-ramp closed form: tau = {tauL*us:.2f} {unit} ({(tauL/tau - 1)*100:+.1f} %), TM = {TML*us:.2f} {unit} "
         f"({(TML/TM - 1)*100:+.1f} %); this set returns Irr = {rL['Irr']:.2f} A ({(rL['Irr']/Irr - 1)*100:+.1f} %), "
         f"Qrr = {rL['Qrr']*1e6:.2f} uC ({(rL['Qrr']/Qrr - 1)*100:+.1f} %) at the datasheet point")
    line(f"   exact set back at the datasheet point: Irr = {r['Irr']:.4f} A, Qrr = {r['Qrr']*1e6:.4f} uC; step-by-step "
         f"integration of the charge equations: Irr = {In:.4f} A, Qrr = {Qn*1e6:.4f} uC")
    line(f"   charge stored at the current zero: {r['q0']*1e6:.1f} uC (tau*IF = {tau*IF*1e6:.1f} uC); recovered "
         f"{Qrr/r['q0']*100:.0f} % of it")
    line(f"   if the datasheet charge is integrated down to {P_END*100:.0f} % of Irr: tau_rr = {trr98*us:.2f} {unit}, "
         f"tau = {tau98*us:.2f} {unit}, TM = {TM98*us:.2f} {unit}")
    line(f"   local exponents of the model at the datasheet point: Irr ~ a^{mI:.2f} IF^{1 - mI:.2f}, "
         f"Qrr ~ a^{mQ:.2f} IF^{1 - mQ:.2f}")
    dg = 1 if us == 1e6 else 0
    put(key + "If", IF, 0); put(key + "Slope", a/1e6, 0); put(key + "Irr", Irr, 0); put(key + "Qrr", Qrr*1e6, 0)
    put(key + "Vr", d["VR"], 0); put(key + "T", d["T"], 0); put(key + "Vrrm", d["VRRM"], 0)
    put(key + "VrrmKv", d["VRRM"]/1e3, 1); put(key + "IrrmMa", d["IRRM"]*1e3, 0)
    put(key + "Ta", ta*us, dg); put(key + "Qtri", 0.5*Irr*ta*1e6, 1 if us == 1e9 else 0)
    put(key + "QtriPct", 0.5*Irr*ta/Qrr*100, 0); put(key + "QoverI", Qrr/Irr*us, dg); put(key + "TaHalf", ta/2*us, dg)
    put(key + "Trr", trr*us, dg); put(key + "S", r["s"], 2); put(key + "Tau", tau*us, dg); put(key + "Tm", TM*us, dg)
    put(key + "TauL", tauL*us, dg); put(key + "TmL", TML*us, dg); put(key + "Rho", r["rho"], 2)
    put(key + "TmOverTau", TM/tau, 2); put(key + "SoftLong", 2*TM/tau, 2); put(key + "IrrOverIf", Irr/IF, 3 if us == 1e6 else 2)
    put(key + "Lam", r["lam"], 2); put(key + "X", r["x"], 2); put(key + "F", r["F"], 3)
    put(key + "Tone", r["T1"]*us, 0); put(key + "Tzero", IF/a*us, 0)
    put(key + "Iter", str(it))
    put(key + "TauLErr", pct(tauL/tau - 1)); put(key + "TmLErr", pct(TML/TM - 1))
    put(key + "IrrBackL", rL["Irr"], 1); put(key + "QrrBackL", rL["Qrr"]*1e6, 1 if us == 1e9 else 0)
    put(key + "IrrBackLErr", pct(rL["Irr"]/Irr - 1)); put(key + "QrrBackLErr", pct(rL["Qrr"]/Qrr - 1))
    put(key + "Qzero", r["q0"]*1e6, 1 if us == 1e9 else 0); put(key + "QzeroPct", Qrr/r["q0"]*100, 0)
    put(key + "Qstat", tau*IF*1e6, 1 if us == 1e9 else 0)
    put(key + "TrrNinetyEight", trr98*us, dg); put(key + "TauNinetyEight", tau98*us, dg)
    put(key + "TmNinetyEight", TM98*us, dg)
    put(key + "ExpIa", mI, 2); put(key + "ExpQa", mQ, 2); put(key + "ExpIf", 1 - mI, 2); put(key + "ExpQf", 1 - mQ, 2)
    put(key + "NumErr", sci(max(abs(In/r["Irr"] - 1), abs(Qn/r["Qrr"] - 1)), 0))
    return dict(a=a, IF=IF, Irr=Irr, Qrr=Qrr, VT=VT, ta=ta, tau_rr=trr, tau=tau, TM=TM, tauL=tauL, TML=TML, rec=r)


def base_width(d, c, key):
    """Plausibility check of TM: width of the base that corresponds to it in the lumped derivation of the model."""
    D, tau, TM = d["D"], c["tau"], c["TM"]
    Ld = math.sqrt(D*tau)                                    # ambipolar diffusion length
    w_par = 2*math.sqrt(3*D*TM)                              # TM = d^2/(3 D), parabolic profile, d = half width
    w_mode = math.pi*math.sqrt(D*TM)                         # TM = 4 d^2/(pi^2 D), slowest mode of the blocked base
    x = brentq(lambda x: x/math.tanh(x) - 1 - TM/tau, 1e-6, 50.0, xtol=1e-12)    # static profile, d/L
    line(f"   base width that corresponds to TM with D = {D*1e4:g} cm2/s: {w_par*1e6:.0f} um (parabolic profile), "
         f"{w_mode*1e6:.0f} um (slowest mode), {2*x*Ld*1e6:.0f} um (static profile, d/L = {x:.2f}); diffusion length "
         f"{Ld*1e6:.0f} um")
    put(key + "Dcm", D*1e4, 0); put(key + "Ldiff", Ld*1e6, 0); put(key + "Wpar", w_par*1e6, 0)
    put(key + "Wmode", w_mode*1e6, 0); put(key + "Wstat", 2*x*Ld*1e6, 0); put(key + "DoverL", x, 2)


def other_sets(d, c, key, us):
    """Equivalent data for other parameter sets and for behavioural models."""
    unit = "us" if us == 1e6 else "ns"
    dg = 1 if us == 1e6 else 0
    a, IF, Irr, Qrr, ta, trr, tau, TM = c["a"], c["IF"], c["Irr"], c["Qrr"], c["ta"], c["tau_rr"], c["tau"], c["TM"]
    k_line, k10 = lm.trr_factor("line"), lm.trr_factor("point", p=0.1)
    t_line, t_10, t_tri = ta + k_line*trr, ta + k10*trr, 2*Qrr/Irr
    L = d["VR"]/a
    line(f"   recovery time: line through 90 % and 25 %: {t_line*us:.1f} {unit}; instant at 10 %: {t_10*us:.1f} {unit} "
         f"(stretch factor {t_10/ta:.2f}); triangle with the same charge 2*Qrr/Irr: {t_tri*us:.1f} {unit}")
    line(f"   fall time over rise time: {k_line*trr/ta:.2f} (line), {2*trr/ta:.2f} (same charge); slope ratio a*tau_rr/Irr "
         f"= {trr/ta:.2f}")
    line(f"   abrupt-commutation estimates: tau*IF = {tau*IF*1e6:.1f} uC against Qrr = {Qrr*1e6:g} uC "
         f"(Qrr/IF = {Qrr/IF*us:.2f} {unit} against tau = {tau*us:.2f} {unit}); limit of the model for a very fast "
         f"commutation: Irr = IF*tau/TM = {IF*tau/TM:.0f} A, Qrr = IF*(tau - tau_rr) = {IF*(tau - trr)*1e6:.1f} uC")
    line(f"   inductive commutation without snubber and capacitance, L = VR/a = {L*1e6:.3g} uH: peak reverse voltage "
         f"VR*(1 + 2/s) = {d['VR']*(1 + 2/c['rec']['s']):.0f} V, energy VR*Qrr = {d['VR']*Qrr*1e3:.2f} mJ")
    put(key + "TrrLine", t_line*us, dg); put(key + "TrrTen", t_10*us, dg); put(key + "TrrTri", t_tri*us, dg)
    put(key + "Stretch", t_10/ta, 2); put(key + "SLine", k_line*trr/ta, 2); put(key + "SlopeRatio", trr/ta, 2)
    put(key + "QoverIf", Qrr/IF*us, 2 if us == 1e6 else 0); put(key + "TauIfRatio", tau*IF/Qrr, 1)
    put(key + "IrrFast", IF*tau/TM, 0); put(key + "QrrFast", IF*(tau - trr)*1e6, 1 if us == 1e9 else 0)
    put(key + "L", L*1e6, 1 if L >= 1e-6 else 2); put(key + "Vpk", d["VR"]*(1 + 2/c["rec"]["s"]), 0)
    put(key + "Erec", d["VR"]*Qrr*1e3, 0 if d["VR"]*Qrr >= 0.1 else 1); put(key + "OvFactor", 1 + 2/c["rec"]["s"], 2)
    # the three inverse maps give the same set: peak and charge, peak and recovery time, charge and recovery time
    t1 = lm.lm_from_irr_trr(a, IF, Irr, t_line, k_line)
    t2 = lm.lm_from_tail(a, IF, lm.irr_from_qrr_trr(a, Qrr, t_10, k10), (t_10 - lm.irr_from_qrr_trr(a, Qrr, t_10, k10)/a)/k10)
    line(f"   inverse maps from (Irr, t_rr line) and from (Qrr, t_rr at 10 %): tau = {t1[0]*us:.3f} and {t2[0]*us:.3f} {unit}, "
         f"TM = {t1[1]*us:.3f} and {t2[1]*us:.3f} {unit}")
    assert abs(t1[0]/tau - 1) < 1e-9 and abs(t2[0]/tau - 1) < 1e-9 and abs(t2[1]/TM - 1) < 1e-9


def sens_table(c, key):
    S = lm.sensitivities(c["a"], c["IF"], c["Irr"], c["Qrr"])
    line("   logarithmic sensitivities of the exact conversion (rows tau, TM, tau_rr; columns Qrr, Irr, a, IF):")
    for name, row in zip(("tau   ", "TM    ", "tau_rr"), S):
        line("     " + name + "  " + "  ".join(f"{v:+7.3f}" for v in row))
    for i, rn in enumerate(("Tau", "Tm", "Trr")):
        for j, cn in enumerate(("Q", "I", "A", "F")):
            put(key + "Sens" + rn + cn, pct(S[i, j]/100, 2))
    dI = 0.05                                                # effect of a 5 % error of the peak current
    t2 = lm.lm_from_irr_qrr(c["a"], c["IF"], c["Irr"]*(1 + dI), c["Qrr"])
    line(f"   peak current higher by {dI*100:.0f} % at the same charge: tau {100*(t2[0]/c['tau'] - 1):+.1f} %, "
         f"TM {100*(t2[1]/c['TM'] - 1):+.1f} %, tau_rr {100*(t2[2]/c['tau_rr'] - 1):+.1f} %")
    put(key + "FiveTau", pct(t2[0]/c["tau"] - 1)); put(key + "FiveTm", pct(t2[1]/c["TM"] - 1, 0))
    put(key + "FiveTrr", pct(t2[2]/c["tau_rr"] - 1, 0))


def cell_check(d, c, fw, key, us):
    """Model with junction capacitance, leakage and series resistance in the simulated commutation cell."""
    unit = "us" if us == 1e6 else "ns"
    dg = 1 if us == 1e6 else 0
    IS, RS, n = fw
    dev = dict(tau=c["tau"], TM=c["TM"], IS=IS, n=n, VT=c["VT"], RS=RS, Rr=d["VRRM"]/d["IRRM"], Cj0=d["CJ0"])
    L = d["VR"]/c["a"]
    s = lm.simulate_cell(dev, c["IF"], d["VR"], L, d["RSN"], d["CSN"])
    m, md = lm.measure(s["t"], s["i"], p_end=P_END), lm.measure(s["t"], s["i_dif"], p_end=P_END)
    r = lm.ramp_recovery(m["a"], c["IF"], c["tau"], c["TM"])
    Q98 = r["Qa"] + (1 - P_END)*r["Qtail"]
    t_line = r["ta"] + lm.trr_factor("line")*r["tau_rr"]
    Qc = lm.cj_charge(d["VR"], d["CJ0"])
    after = s["t"] > m["t0"]
    E_d = lm.trapezoid(s["v"]*s["i"]*after, s["t"])
    E_r = lm.trapezoid(d["RSN"]*s["i_sn"]**2, s["t"])
    line(f"   test cell: L = VR/a = {L*1e6:.3g} uH, snubber {d['RSN']:g} Ohm + {d['CSN']*1e9:g} nF, Cj0 = {d['CJ0']*1e9:g} nF, "
         f"Rr = {dev['Rr']/1e3:.0f} kOhm")
    line(f"   slope at the current zero {m['a']/1e6:.4g} A/us ({(m['a']/c['a'] - 1)*100:+.2f} % on VR/L, forward voltage)")
    line(f"   peak reverse current: closed form at that slope {r['Irr']:.2f} A; simulation, diffusion part "
         f"{md['Irr']:.2f} A ({(md['Irr']/r['Irr'] - 1)*100:+.2f} %), at the terminals {m['Irr']:.2f} A "
         f"({(m['Irr']/r['Irr'] - 1)*100:+.2f} %)")
    line(f"   charge down to {P_END*100:.0f} % of the peak: closed form {Q98*1e6:.3f} uC; simulation, diffusion part "
         f"{md['Qrr']*1e6:.3f} uC ({(md['Qrr']/Q98 - 1)*100:+.2f} %), at the terminals {m['Qrr']*1e6:.3f} uC "
         f"({(m['Qrr']/Q98 - 1)*100:+.2f} %); charge of the junction capacitance up to VR: {Qc*1e6:.4f} uC "
         f"({Qc/c['Qrr']*100:.2f} % of Qrr)")
    line(f"   recovery time by the line through 90 % and 25 %: closed form {t_line*us:.2f} {unit}, simulation "
         f"{m['trr_line']*us:.2f} {unit} ({(m['trr_line']/t_line - 1)*100:+.2f} %)")
    E_c = 0.5*d["CSN"]*d["VR"]**2
    E_bal = d["VR"]*r["Qrr"] + E_c
    line(f"   peak reverse voltage {-s['v'].min():.0f} V; energy in the diode after the current zero {E_d*1e3:.3f} mJ, in "
         f"the snubber resistor {E_r*1e3:.3f} mJ, sum {(E_d + E_r)*1e3:.3f} mJ; balance VR*Qrr + Csn*VR^2/2 = "
         f"{d['VR']*r['Qrr']*1e3:.3f} + {E_c*1e3:.3f} = {E_bal*1e3:.3f} mJ ({((E_d + E_r)/E_bal - 1)*100:+.2f} %)")
    put(key + "Cj", d["CJ0"]*1e9, 0); put(key + "Rsn", d["RSN"], 1 if d["RSN"] % 1 else 0)
    put(key + "Csn", d["CSN"]*1e9, 0); put(key + "CsnUf", d["CSN"]*1e6, 2)
    put(key + "RsnCsn", d["RSN"]*d["CSN"]*us, dg)
    put(key + "SimSlope", m["a"]/1e6, 2 if us == 1e6 else 0); put(key + "SimSlopeErr", pct(m["a"]/c["a"] - 1, 2))
    put(key + "SimIrrCf", r["Irr"], 2); put(key + "SimIrrDif", md["Irr"], 2); put(key + "SimIrrTerm", m["Irr"], 2)
    put(key + "SimIrrDifErr", pct(md["Irr"]/r["Irr"] - 1, 2)); put(key + "SimIrrTermErr", pct(m["Irr"]/r["Irr"] - 1, 2))
    qd = 0 if us == 1e6 else 2
    put(key + "SimQCf", Q98*1e6, qd); put(key + "SimQDif", md["Qrr"]*1e6, qd); put(key + "SimQTerm", m["Qrr"]*1e6, qd)
    put(key + "SimQDifErr", pct(md["Qrr"]/Q98 - 1, 2)); put(key + "SimQTermErr", pct(m["Qrr"]/Q98 - 1, 2))
    put(key + "Qc", Qc*1e6, 2 if us == 1e6 else 3); put(key + "QcPct", Qc/c["Qrr"]*100, 2)
    put(key + "SimTrrCf", t_line*us, dg + 1); put(key + "SimTrr", m["trr_line"]*us, dg + 1)
    put(key + "SimTrrErr", pct(m["trr_line"]/t_line - 1, 2))
    put(key + "SimVpk", -s["v"].min(), 0); put(key + "SimEd", E_d*1e3, 0 if E_d > 0.1 else 2)
    put(key + "SimEr", E_r*1e3, 0 if E_d > 0.1 else 2)
    put(key + "SimEsum", (E_d + E_r)*1e3, 0 if E_d > 0.1 else 2)
    put(key + "SimEcf", d["VR"]*r["Qrr"]*1e3, 0 if E_d > 0.1 else 2)
    put(key + "SimEc", E_c*1e3, 1 if E_d > 0.1 else 2); put(key + "SimEbal", E_bal*1e3, 0 if E_d > 0.1 else 2)
    put(key + "SimEbalErr", pct((E_d + E_r)/E_bal - 1, 1))
    put(key + "SimOv", -s["v"].min()/d["VR"], 2)
    return dict(s=s, m=m, md=md, r=r, dev=dev)


def away(d, c, key, us, slopes, currents):
    """The model away from the datasheet point: other slopes at the datasheet current, other currents at the
    datasheet slope."""
    a0, IF0 = c["a"], c["IF"]
    rows = []
    line("   model away from the datasheet point (slope in A/us, IF in A, Irr in A, Qrr in uC):")
    for k, f in enumerate(slopes):
        r = lm.ramp_recovery(a0*f, IF0, c["tau"], c["TM"])
        ds = d["QRR"]*f**d["Q_EXP"] if "Q_EXP" in d else None
        rows.append((a0*f, IF0, r, ds))
        line(f"     a = {a0*f/1e6:8.3g}, IF = {IF0:6.0f}: Irr = {r['Irr']:8.3f}, Qrr = {r['Qrr']*1e6:9.3f}, T1/tau = {r['x']:.2f}"
             + (f"; datasheet curve {ds*1e6:.0f} uC (model/curve = {r['Qrr']/ds:.2f})" if ds else ""))
        tag = ("Lo", "LoB", "Ref", "HiB", "Hi")[k] if len(slopes) == 5 else ("Lo", "Ref", "Hi")[k]
        qd = 0 if us == 1e6 else 1
        put(key + "AwayA" + tag, f"{a0*f/1e6:g}"); put(key + "AwayIrrA" + tag, r["Irr"], 1 if r["Irr"] < 100 else 0)
        put(key + "AwayQrrA" + tag, r["Qrr"]*1e6, qd); put(key + "AwayXA" + tag, r["x"], 2)
        if ds:
            put(key + "AwayDsA" + tag, ds*1e6, 0); put(key + "AwayRatioA" + tag, r["Qrr"]/ds, 2)
            # the point of the datasheet curve at this slope converted on its own, with the softness of the
            # datasheet point: the set that the curve asks for at this slope
            Ic = lm.irr_from_qrr_softness(a0*f, ds, c["rec"]["s"])
            tc, Tc, _ = lm.lm_from_irr_qrr(a0*f, IF0, Ic, ds)
            line(f"       curve point converted with the same softness: Irr = {Ic:.1f} A, tau = {tc*us:.1f}, TM = {Tc*us:.1f}")
            put(key + "AwayIcA" + tag, Ic, 1 if Ic < 100 else 0); put(key + "AwayTauA" + tag, tc*us, 1)
            put(key + "AwayTmA" + tag, Tc*us, 1)
    for k, f in enumerate(currents):
        r = lm.ramp_recovery(a0, IF0*f, c["tau"], c["TM"])
        line(f"     a = {a0/1e6:8.3g}, IF = {IF0*f:6.0f}: Irr = {r['Irr']:8.3f}, Qrr = {r['Qrr']*1e6:9.3f}, T1/tau = {r['x']:.2f}")
        tag = ("Lo", "Ref", "Hi")[k]
        qd = 0 if us == 1e6 else 1
        put(key + "AwayF" + tag, IF0*f, 0); put(key + "AwayIrrF" + tag, r["Irr"], 1 if r["Irr"] < 100 else 0)
        put(key + "AwayQrrF" + tag, r["Qrr"]*1e6, qd)
    return rows


def section_a():
    line("2. Diode A: rectifier diode, long ramp")
    d = A
    c = convert(d, "A", 1e6)
    VT, n = c["VT"], d["N"]
    # forward characteristic: straight line of the datasheet, secant and tangent fits for n = 1 and n = 2
    fits = {}
    for nn in (1.0, 2.0):
        ISs, RSs = lm.forward_secant(d["VT0"], d["RT"], d["I1"], d["I2"], nn, VT)
        ISt, RSt = lm.forward_tangent(d["VT0"], d["RT"], d["IF"], nn, VT)
        fits[nn] = (ISs, RSs, ISt, RSt)
        line(f"   forward characteristic, n = {nn:g}: secant through {d['I1']:g} A and {d['I2']:g} A: IS = {ISs:.3e} A, "
             f"RS = {RSs*1e3:.4f} mOhm; tangent at {d['IF']:g} A: IS = {ISt:.3e} A, RS = {RSt*1e3:.4f} mOhm")
        tag = "One" if nn == 1.0 else "Two"
        put("AIsSec" + tag, sci(ISs)); put("ARsSec" + tag, RSs*1e3, 3)
        put("AIsTan" + tag, sci(ISt)); put("ARsTan" + tag, RSt*1e3, 3)
    IS, RS = fits[n][0], fits[n][1]
    i_chk = np.array([50.0, 100.0, 250.0, d["I1"], d["IF"], d["I2"], 2500.0])
    v_line = d["VT0"] + d["RT"]*i_chk
    v_mod = lm.forward_voltage(i_chk, IS, n, VT, RS)
    line("   model (secant, n = %g) against the straight line: " % n
         + ", ".join(f"{i:g} A: {vm:.3f} V ({(vm - vl)*1e3:+.0f} mV)" for i, vm, vl in zip(i_chk, v_mod, v_line)))
    put("AVt", VT*1e3, 2); put("AVto", d["VT0"], 2); put("ARt", d["RT"]*1e3, 2); put("AIone", d["I1"], 0)
    put("AItwo", d["I2"], 0); put("AN", n, 0)
    put("ANmaxSec", d["RT"]*(d["I2"] - d["I1"])/(VT*math.log(d["I2"]/d["I1"])), 1)
    put("ANmaxTan", d["RT"]*d["IF"]/VT, 1)
    put("ASlopeJct", n*VT/d["IF"]*1e3, 3)
    put("AVfLine", d["VT0"] + d["RT"]*d["IF"], 2)
    put("AVlow", float(lm.forward_voltage(100.0, IS, n, VT, RS)), 2); put("AVlowLine", d["VT0"] + d["RT"]*100.0, 2)
    put("AVlowDiff", (d["VT0"] + d["RT"]*100.0 - float(lm.forward_voltage(100.0, IS, n, VT, RS)))*1e3, 0)
    ISo, RSo = fits[1.0][0], fits[1.0][1]
    put("AVlowOne", float(lm.forward_voltage(100.0, ISo, 1.0, VT, RSo)), 2)
    put("AIsStar", sci(IS*(c["tau"] + c["TM"])/c["tau"])); put("AIsFactor", (c["tau"] + c["TM"])/c["tau"], 2)
    put("AIsOffsetMv", n*VT*math.log((c["tau"] + c["TM"])/c["tau"])*1e3, 0)
    # blocking
    Rr = d["VRRM"]/d["IRRM"]
    line(f"   leakage at the rated point: Rr = VRRM/IRRM = {Rr/1e3:.0f} kOhm; charge of the junction capacitance up to "
         f"VR = {d['VR']:g} V: {lm.cj_charge(d['VR'], d['CJ0'])*1e6:.3f} uC")
    put("ARr", Rr/1e3, 0)
    base_width(d, c, "A")
    other_sets(d, c, "A", 1e6)
    sens_table(c, "A")
    chk = cell_check(d, c, (IS, RS, n), "A", 1e6)
    rows = away(d, c, "A", 1e6, (0.1, 0.5, 1.0, 2.0, 10.0), (0.5, 1.0, 2.0))
    put("AQexp", d["Q_EXP"], 2); put("AQrangeLo", d["Q_RANGE"][0]/1e6, 1); put("AQrangeHi", d["Q_RANGE"][1]/1e6, 0)
    put("ATauExpCurve", (1 - d["Q_EXP"])/2, 2)              # long ramp, same softness: tau ~ a^(-(1 - m)/2)
    # second temperature with the law tau ~ T (absolute temperature), same softness
    kT = (d["T2"] + 273.15)/(d["T"] + 273.15)
    r2 = lm.ramp_recovery(c["a"], c["IF"], c["tau"]*kT, c["TM"]*kT)
    line(f"   at {d['T2']:g} C with tau and TM scaled by the ratio of the absolute temperatures ({kT:.3f}, assumed): tau = {c['tau']*kT*1e6:.1f} us, "
         f"TM = {c['TM']*kT*1e6:.1f} us; at the datasheet slope and current Irr = {r2['Irr']:.1f} A "
         f"({(r2['Irr']/c['Irr'] - 1)*100:+.0f} %), Qrr = {r2['Qrr']*1e6:.0f} uC ({(r2['Qrr']/c['Qrr'] - 1)*100:+.0f} %)")
    put("ATtwo", d["T2"], 0); put("AKt", kT, 3); put("ATauTtwo", c["tau"]*kT*1e6, 1); put("ATmTtwo", c["TM"]*kT*1e6, 1)
    put("AIrrTtwo", r2["Irr"], 0); put("AQrrTtwo", r2["Qrr"]*1e6, 0)
    put("AIrrTtwoPct", (1 - r2["Irr"]/c["Irr"])*100, 0); put("AQrrTtwoPct", (1 - r2["Qrr"]/c["Qrr"])*100, 0)
    # half-sine test pulse with the same slope at the current zero
    for Th, tag in ((10e-3, "Long"), (1e-3, "Short")):
        w = math.pi/Th
        Ipk = c["a"]/w
        Is_, phi = lm.sine_peak(Ipk, w, c["tau"], c["TM"])
        ref = c["a"]*(c["tau"] - c["tau_rr"])
        line(f"   half sine of {Th*1e3:g} ms with {c['a']/1e6:g} A/us at the zero (peak {Ipk:.0f} A): w*tau = {w*c['tau']:.4f}, "
             f"Irr = {Is_:.2f} A ({(Is_/ref - 1)*100:+.2f} % on a*(tau - tau_rr) = {ref:.2f} A)")
        put("ASine" + tag + "Ms", Th*1e3, 0); put("ASine" + tag + "Ipk", Ipk/1e3, 1 if Ipk >= 1e4 else 2)
        put("ASine" + tag + "Wt", w*c["tau"], 3); put("ASine" + tag + "Err", pct(Is_/ref - 1, 2))
    put("ALongIrr", c["a"]*(c["tau"] - c["tau_rr"]), 1)
    line()
    return dict(d=d, c=c, fits=fits, IS=IS, RS=RS, n=n, chk=chk, rows=rows)


def section_b():
    line("3. Diode B: fast diode, short ramp")
    d = B
    c = convert(d, "B", 1e9)
    VT = c["VT"]
    # forward characteristic from three points of the curve
    IS, RS, n = lm.forward_points(d["VF"], VT)
    (I1, V1), (I2, V2), (I3, V3) = d["VF"]
    rT = (V3 - V2)/(I3 - I2)
    VT0 = V3 - rT*I3
    line(f"   forward characteristic through {', '.join(f'{v:g} V at {i:g} A' for i, v in d['VF'])}: n = {n:.3f}, "
         f"RS = {RS*1e3:.3f} mOhm, IS = {IS:.3e} A (VT = {VT*1e3:.2f} mV); straight line through the two upper points: "
         f"VT0 = {VT0:.2f} V, rT = {rT*1e3:.1f} mOhm")
    chk_v = lm.forward_voltage(np.array([I1, I2, I3]), IS, n, VT, RS)
    assert np.allclose(chk_v, [V1, V2, V3], atol=1e-6)
    put("BVt", VT*1e3, 2); put("BN", n, 2); put("BRs", RS*1e3, 1); put("BIs", sci(IS)); put("BNvt", n*VT*1e3, 1)
    for tag, (i, v) in zip(("One", "Two", "Three"), d["VF"]):
        put("BIf" + tag, i, 0); put("BVf" + tag, v, 2)
    put("BVto", VT0, 2); put("BRt", rT*1e3, 0)
    put("BIsStar", sci(IS*(c["tau"] + c["TM"])/c["tau"])); put("BIsFactor", (c["tau"] + c["TM"])/c["tau"], 2)
    put("BIsOffsetMv", n*VT*math.log((c["tau"] + c["TM"])/c["tau"])*1e3, 0)
    Rr = d["VRRM"]/d["IRRM"]
    put("BRr", Rr/1e6, 1)
    line(f"   leakage at the rated point: Rr = {Rr/1e6:.1f} MOhm; charge of the junction capacitance up to VR = {d['VR']:g} V: "
         f"{lm.cj_charge(d['VR'], d['CJ0'])*1e9:.1f} nC")
    base_width(d, c, "B")
    other_sets(d, c, "B", 1e9)
    sens_table(c, "B")
    chk = cell_check(d, c, (IS, RS, n), "B", 1e9)
    rows = away(d, c, "B", 1e9, (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    # second temperature: the datasheet gives the recovery data again, so the set is extracted again
    tau2, TM2, trr2 = lm.lm_from_irr_qrr(c["a"], c["IF"], d["IRR2"], d["QRR2"])
    line(f"   at {d['T2']:g} C from Irr = {d['IRR2']:g} A, Qrr = {d['QRR2']*1e6:g} uC: tau_rr = {trr2*1e9:.1f} ns, "
         f"tau = {tau2*1e9:.1f} ns ({(tau2/c['tau'] - 1)*100:+.0f} %), TM = {TM2*1e9:.1f} ns ({(TM2/c['TM'] - 1)*100:+.0f} %)")
    kT = (d["T2"] + 273.15)/(d["T"] + 273.15)
    put("BTtwo", d["T2"], 0); put("BIrrTtwo", d["IRR2"], 0); put("BQrrTtwo", d["QRR2"]*1e6, 0)
    put("BTrrTtwo", trr2*1e9, 0); put("BTauTtwo", tau2*1e9, 0); put("BTmTtwo", TM2*1e9, 0)
    put("BTauTtwoPct", (1 - tau2/c["tau"])*100, 0); put("BTmTtwoPct", (1 - TM2/c["TM"])*100, 0)
    put("BTauExp", math.log(tau2/c["tau"])/math.log(kT), 1); put("BKt", kT, 3)
    line(f"   equivalent exponent of the lifetime against the absolute temperature: {math.log(tau2/c['tau'])/math.log(kT):.2f}")
    # resistive test with small currents
    IFs, IRs, Irec = d["STEP"]
    ts, I0, Q = lm.step_recovery(IFs, IRs, c["tau"], c["TM"])
    t_read = ts + c["tau_rr"]*math.log(I0/Irec)
    line(f"   resistive test, IF = {IFs:g} A, IR = {IRs:g} A, read at {Irec:g} A: IF/IR = {IFs/IRs:g} against TM/tau = "
         f"{c['TM']/c['tau']:.3f}; storage time {ts*1e9:.1f} ns, current at the start of the tail {I0:.3f} A, time to "
         f"{Irec:g} A: {t_read*1e9:.0f} ns, charge {Q*1e9:.1f} nC")
    put("BStepIf", IFs, 1); put("BStepIr", IRs, 0); put("BStepIrec", Irec, 2); put("BStepIzero", I0, 2)
    put("BStepTs", ts*1e9, 0); put("BStepTrr", t_read*1e9, 0); put("BStepQ", Q*1e9, 0)
    put("BStepRatio", IFs/IRs, 1)
    # with a higher forward current in the same test there is a storage phase
    IFs2 = 2.0
    ts2, I02, Q2 = lm.step_recovery(IFs2, IRs, c["tau"], c["TM"])
    line(f"   same test with IF = {IFs2:g} A: storage time {ts2*1e9:.0f} ns, time to {Irec:g} A "
         f"{(ts2 + c['tau_rr']*math.log(I02/Irec))*1e9:.0f} ns")
    put("BStepIfTwo", IFs2, 0); put("BStepTsTwo", ts2*1e9, 0)
    put("BStepTrrTwo", (ts2 + c["tau_rr"]*math.log(I02/Irec))*1e9, 0)
    line()
    return dict(d=d, c=c, IS=IS, RS=RS, n=n, chk=chk, rows=rows, VT0=VT0, rT=rT)


# ---------------------------------------------------------------- figures
# Each figure is drawn twice: English labels for lauritzen_ma_datasheet_conversion.tex (figures/fig_*.pdf) and
# Italian labels for lauritzen_ma_datasheet_conversion_it.tex (figures/fig_*_it.pdf). L(en, it) returns the label
# in the current language.
LANG = "en"


def L(en, it):
    return it if LANG == "it" else en


def loc_num(s):
    """A number written in a label: decimal point in English, decimal comma in Italian."""
    return s.replace(".", ",") if LANG == "it" else s


def loc_math(s):
    """The same inside mathtext, where the comma is grouped so that it takes no punctuation space."""
    return s.replace(".", "{,}") if LANG == "it" else s


def comma_ticks(fig):
    """Italian figures: decimal comma in the labels of the major ticks (limits kept)."""
    fig.canvas.draw()
    for ax in fig.axes:
        lims = ax.get_xlim(), ax.get_ylim()
        for axis in (ax.xaxis, ax.yaxis):
            labs = [t.get_text() for t in axis.get_majorticklabels()]
            if any("." in v for v in labs):
                axis.set_ticks(axis.get_majorticklocs())
                axis.set_ticklabels([v.replace(".", ",") for v in labs])
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    if LANG == "it":
        comma_ticks(fig)
    fig.savefig(os.path.join(FIGDIR, name + ("_it" if LANG == "it" else "") + ".pdf"))
    plt.close(fig)


def note(ax, text, xy, xytext, ha="left", va="center", arrow=True, color=None):
    ax.annotate(text, xy, xytext=xytext, textcoords="offset points", ha=ha, va=va, fontsize=7, color=color or INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2) if arrow else None)


def end_labels(ax, x, ys, labels, sep=8.0, dx=3):
    """Direct labels at the right end of the curves, moved apart to at least `sep` points (limits set before)."""
    pts = [ax.transData.transform((x, y))[1]*72.0/ax.figure.dpi for y in ys]
    order = sorted(range(len(ys)), key=lambda k: pts[k])
    pos = [pts[k] for k in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + sep)
    shift = 0.5*((pos[-1] - pts[order[-1]]) + (pos[0] - pts[order[0]]))      # keep the group centred
    for k, j in enumerate(order):
        ax.annotate(labels[j], (x, ys[j]), xytext=(dx, pos[k] - shift - pts[j]), textcoords="offset points",
                    fontsize=7, va="center", ha="left", annotation_clip=False)


def plain_log(ax, which="both"):
    """Logarithmic axes with plain numbers at 1 and 3 per decade."""
    fmt = mticker.FuncFormatter(lambda v, p: f"{v:g}")
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        if which in ("both", name):
            axis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 3.0), numticks=12))
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(mticker.NullFormatter())


def fig_waveform():
    """Turn-off on a ramp: current with the constructions of the datasheet quantities, and the two charges."""
    tau, rho, lam = 1.0, 0.35, 2.2
    TM = rho*tau/(1 - rho)
    a, IF = 1.0, lam
    r = lm.ramp_recovery(a, IF, tau, TM)
    t0, T1, Irr, trr, ta = IF/a, r["T1"], r["Irr"], r["tau_rr"], r["ta"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.28))
    # --- current, time from the current zero in units of tau, current in units of Irr
    t = np.linspace(t0 - 0.75, T1 + 5.2*trr, 3000)
    i, qM, qE = lm.ramp_waveform(t, a, IF, tau, TM)
    x, y = t - t0, i/Irr
    b = ax[0]
    tri = (x >= 0) & (x <= ta)
    tail = x >= ta
    b.fill_between(x[tri], y[tri], 0, color=RAMP[0], alpha=0.55, lw=0)
    b.fill_between(x[tail], y[tail], 0, color=C_4, alpha=0.35, lw=0)
    b.axhline(0, color=AXIS, lw=0.6)
    b.plot(x, y, color=C_NOM)
    kA, kB = -math.log(0.9), -math.log(0.25)
    xz = ta + lm.trr_factor("line")*trr
    b.plot([ta + kA*trr, xz], [-0.9, 0.0], color=INK, lw=0.8, ls=(0, (4, 2)))
    b.plot([ta + kA*trr, ta + kB*trr], [-0.9, -0.25], "o", ms=3, color=INK)
    x10 = ta + math.log(10)*trr
    b.plot([x10], [-0.1], "s", ms=3, color=C_WEAK)
    b.text(ta + kA*trr - 0.03, -0.92, "A", fontsize=7, ha="right", va="top")
    b.text(ta + kB*trr + 0.035, -0.27, "B", fontsize=7, ha="left", va="top")
    note(b, L("10 % of", "10 % di") + " $I_\\mathrm{rr}$", (x10, -0.1), (10, -16), color=INK)
    note(b, "$I_\\mathrm{rr}\\,t_\\mathrm{a}/2$", (0.62*ta, -0.3), (-24, -26), ha="right")
    note(b, "$I_\\mathrm{rr}\\,\\tau_\\mathrm{rr}$", (ta + 0.55*trr, -0.28), (26, -30))
    b.annotate("", (0, 0.2), (ta, 0.2), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6, shrinkA=0, shrinkB=0))
    b.text(ta/2, 0.24, "$t_\\mathrm{a}$", ha="center", va="bottom", fontsize=7.5)
    b.annotate("", (ta, 0.2), (xz, 0.2), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6, shrinkA=0, shrinkB=0))
    b.text((ta + xz)/2, 0.24, loc_math("$t_\\mathrm{b} = %.2f\\,\\tau_\\mathrm{rr}$" % lm.trr_factor("line")), ha="center", va="bottom",
           fontsize=7.5)
    b.plot([ta, ta], [-1, 0.2], color=MUTED, lw=0.5, ls=":")
    b.plot([xz, xz], [0, 0.2], color=MUTED, lw=0.5, ls=":")
    b.text(-0.42, 0.33, L("slope", "di/dt") + " $-a$", fontsize=7, color=INK2, ha="left", va="bottom", rotation=-48)
    b.text(ta + 0.02, -1.04, "$-I_\\mathrm{rr}$", fontsize=7.5, ha="left", va="top")
    b.set_xlim(x[0], x[-1]); b.set_ylim(-1.18, 0.75)
    b.set_xlabel(L("time from the current zero", "tempo dallo zero della corrente") + ", $t/\\tau$")
    b.set_ylabel(L("diode current", "corrente del diodo") + ", $i/I_\\mathrm{rr}$")
    b.set_title(L("(a) current and datasheet quantities", "(a) corrente e grandezze del datasheet"), loc="left")
    # --- charges, in units of a*tau^2
    b = ax[1]
    t = np.linspace(-0.6, T1 + 5.2*trr, 3000)
    i, qM, qE = lm.ramp_waveform(t, a, IF, tau, TM)
    x = t - t0
    b.axhline(0, color=AXIS, lw=0.6)
    b.plot(x, qE, color=C_WEAK)
    b.plot(x, qM, color=C_NOM)
    b.plot(x, tau*i, color=INK2, lw=0.8, ls=(0, (4, 2)))
    b.axvline(0, color=MUTED, lw=0.5, ls=":")
    b.axvline(ta, color=MUTED, lw=0.5, ls=":")
    k = int(np.argmin(np.abs(x + 1.75)))
    note(b, "$q_\\mathrm{E}$", (x[k], qE[k]), (10, 6))
    note(b, "$q_\\mathrm{M}$", (x[k], qM[k]), (10, 8))
    k = int(np.argmin(np.abs(x + 0.9)))
    note(b, "$\\tau\\,i$", (x[k], tau*i[k]), (-16, -14), ha="right")
    b.text(0.03, 0.97, L("current\nzero", "zero della\ncorrente"), fontsize=7, color=INK2, ha="left", va="top", transform=b.get_xaxis_transform())
    b.text(ta + 0.03, 0.80, L("junction\nblocks", "giunzione\ninterdetta"), fontsize=7, color=INK2, ha="left", va="top", transform=b.get_xaxis_transform())
    b.set_xlim(x[0], x[-1])
    b.set_xlabel(L("time from the current zero", "tempo dallo zero della corrente") + ", $t/\\tau$")
    b.set_ylabel(L("charge", "carica") + ", $q/(a\\tau^2)$")
    b.set_title(L("(b) charges of the model", "(b) cariche del modello"), loc="left")
    save(fig, "fig_waveform")
    put("FigRho", rho, 2); put("FigLam", lam, 1)


def fig_map(exA, exB):
    """Normalised recovery of the model against the normalised slope."""
    rhos = (0.2, 0.35, 0.5)
    cols = (RAMP[1], RAMP[3], RAMP[5])
    u = np.logspace(-2, 2, 400)                              # a*tau/IF = 1/lam
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.55), gridspec_kw=dict(wspace=0.50))
    ends = [[], []]
    for rho, col in zip(rhos, cols):
        y = lm.ramp_peak_norm(1/u, rho)
        irr, qrr = y*u, (0.5*y*y + rho*y)*u
        e = (1/u)*(1 - rho - y)
        mq = 1 - e/(0.5*y*y + rho*y)
        ax[0].plot(u, irr, color=col); ax[1].plot(u, qrr, color=col); ax[2].plot(u, mq, color=col)
        ends[0].append(irr[-1]); ends[1].append(qrr[-1])
    rho = rhos[1]                                            # asymptotes of the middle curve
    for b, lo, hi in ((ax[0], (1 - rho)*u, (1 - rho)/rho + 0*u), (ax[1], 0.5*(1 - rho**2)*u, (1 - rho) + 0*u)):
        b.plot(u[lo < 1.5*hi], lo[lo < 1.5*hi], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        b.plot(u, hi, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
    for b in ax[:2]:
        b.set_xscale("log"); b.set_yscale("log")
    ax[2].set_xscale("log")
    ax[0].set_ylim(5e-3, 8); ax[1].set_ylim(3e-3, 3); ax[2].set_ylim(0, 1.08)
    for b in ax:
        b.set_xlim(1e-2, 1e2); b.set_xlabel(L("normalised slope", "di/dt normalizzato") + ", $a\\tau/I_\\mathrm{F}$")
        b.set_xticks([1e-2, 1e0, 1e2])
    end_labels(ax[0], u[-1], ends[0], [loc_num(f"{r:g}") for r in rhos])
    end_labels(ax[1], u[-1], ends[1], [loc_num(f"{r:g}") for r in rhos])
    ax[0].text(0.55, 0.012, L("long ramp:\nslope 1", "rampa lunga:\npendenza 1"), fontsize=7, color=INK2, ha="left", va="center")
    ax[0].text(0.012, 2.7, L("abrupt", "brusca") + ": $I_\\mathrm{F}\\tau/T_\\mathrm{M}$", fontsize=7, color=INK2, ha="left", va="center")
    ax[1].text(0.012, 1.25, L("abrupt", "brusca") + ": $I_\\mathrm{F}(\\tau-\\tau_\\mathrm{rr})$", fontsize=7, color=INK2, ha="left",
               va="center")
    ax[1].text(0.55, 0.0065, L("long ramp:\nslope 1", "rampa lunga:\npendenza 1"), fontsize=7, color=INK2, ha="left", va="center")
    for ex, lab in ((exA, "A"), (exB, "B")):
        c = ex["c"]
        uu = c["a"]*c["tau"]/c["IF"]
        mI, mQ = lm.ramp_exponents(c["a"], c["IF"], c["tau"], c["TM"])
        up, dn = ((-6, 4), "right", "bottom"), ((6, -4), "left", "top")
        for b, yy, (off, ha, va) in ((ax[0], c["Irr"]/c["IF"], up if lab == "A" else dn),
                                     (ax[1], c["Qrr"]/(c["tau"]*c["IF"]), up if lab == "A" else dn),
                                     (ax[2], mQ, ((-6, -5), "right", "top") if lab == "A" else ((6, 3), "left", "bottom"))):
            b.plot([uu], [yy], "o", ms=3.5, color=C_WEAK, zorder=5)
            b.annotate(lab, (uu, yy), xytext=off, textcoords="offset points", fontsize=7.5, ha=ha, va=va, color=INK)
    ax[0].set_ylabel("$I_\\mathrm{rr}/I_\\mathrm{F}$"); ax[1].set_ylabel("$Q_\\mathrm{rr}/(\\tau I_\\mathrm{F})$")
    ax[2].set_ylabel("$\\partial\\ln Q_\\mathrm{rr}/\\partial\\ln a$")
    ax[0].set_title(L("(a) peak reverse current", "(a) peak reverse current"), loc="left")
    ax[1].set_title(L("(b) recovered charge", "(b) recovered charge"), loc="left")
    ax[2].set_title(L("(c) exponent of the charge", "(c) esponente della carica"), loc="left")
    ax[2].text(0.97, 0.97, L("$\\rho$ = 0.2,\n0.35, 0.5\n(light to dark)", "$\\rho$ = 0,2;\n0,35; 0,5\n(da chiaro\na scuro)"), transform=ax[2].transAxes, fontsize=7,
               ha="right", va="top", color=INK2)
    save(fig, "fig_map")


def fig_inversion(exA, exB):
    """What the long-ramp closed form returns at the datasheet point, against Irr/IF of the data."""
    ss = (0.5, 1.0, 2.0, 4.0)
    cols = (RAMP[0], RAMP[2], RAMP[4], RAMP[5])
    q = np.logspace(math.log10(0.03), math.log10(1.0), 120)
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.16))
    ends = [[], []]
    for s, col in zip(ss, cols):
        res = np.array([closed_form_back(1/v, s) for v in q])
        ax[0].plot(q, (res[:, 0] - 1)*100, color=col)
        ax[1].plot(q, (res[:, 1] - 1)*100, color=col)
        ends[0].append((res[-1, 0] - 1)*100); ends[1].append((res[-1, 1] - 1)*100)
    for b in ax:
        b.set_xscale("log"); b.set_xlim(q[0], q[-1]); b.set_ylim(-80, 4)
        b.set_xlabel(L("data", "dati") + ": $I_\\mathrm{rr}/I_\\mathrm{F}$")
        b.axhline(0, color=AXIS, lw=0.6)
        b.set_xticks([0.03, 0.1, 0.3, 1.0]); b.set_xticklabels([loc_num(v) for v in ("0.03", "0.1", "0.3", "1")])
    end_labels(ax[0], q[-1], ends[0], [loc_num(f"{v:g}") for v in ss])
    end_labels(ax[1], q[-1], ends[1], [loc_num(f"{v:g}") for v in ss])
    for ex, lab in ((exA, "A"), (exB, "B")):
        c = ex["c"]
        v = c["Irr"]/c["IF"]
        iI, qQ, _, _ = closed_form_back(1/v, c["rec"]["s"])
        for b, yy in ((ax[0], (iI - 1)*100), (ax[1], (qQ - 1)*100)):
            b.plot([v], [yy], "o", ms=3.5, color=C_WEAK, zorder=5)
            b.annotate(lab, (v, yy), xytext=(-5, -4), textcoords="offset points", fontsize=7.5, ha="right", va="top")
    ax[0].set_ylabel(L("error of the returned value (%)", "errore del valore restituito (%)"))
    ax[0].set_title(L("(a) peak reverse current", "(a) peak reverse current"), loc="left")
    ax[1].set_title(L("(b) recovered charge", "(b) recovered charge"), loc="left")
    ax[0].text(0.04, 0.08, L("softness $s$ = 0.5, 1, 2, 4\n(light to dark)", "softness $s$ = 0,5; 1; 2; 4\n(da chiaro a scuro)"), transform=ax[0].transAxes, fontsize=7,
               ha="left", va="bottom", color=INK2)
    save(fig, "fig_inversion")


def fig_forward(exA, exB):
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.27))
    d, c = exA["d"], exA["c"]
    i = np.linspace(1.0, 2000.0, 800)
    b = ax[0]
    b.plot(i, d["VT0"] + d["RT"]*i, color=INK, lw=0.9, ls=(0, (4, 2)))
    for nn, col in ((1.0, C_WEAK), (2.0, C_NOM)):
        ISs, RSs = exA["fits"][nn][0], exA["fits"][nn][1]
        b.plot(i, lm.forward_voltage(i, ISs, nn, c["VT"], RSs), color=col)
    b.plot([d["I1"], d["I2"]], [d["VT0"] + d["RT"]*d["I1"], d["VT0"] + d["RT"]*d["I2"]], "o", ms=3.5, color=INK)
    k = int(np.argmin(np.abs(i - 130)))
    note(b, "$n$ = 1", (i[k], float(lm.forward_voltage(i[k], exA["fits"][1.0][0], 1.0, c["VT"], exA["fits"][1.0][1]))),
         (16, -12))
    k = int(np.argmin(np.abs(i - 60)))
    note(b, "$n$ = 2", (i[k], float(lm.forward_voltage(i[k], exA["fits"][2.0][0], 2.0, c["VT"], exA["fits"][2.0][1]))),
         (22, -22))
    note(b, "$V_\\mathrm{T0} + r_\\mathrm{T}\\,i$", (230, d["VT0"] + d["RT"]*230), (6, 40), va="bottom")
    b.set_xlim(0, 2000); b.set_ylim(0.5, 2.1)
    b.set_xlabel(L("forward current (A)", "forward current (A)")); b.set_ylabel(L("forward voltage (V)", "forward voltage (V)"))
    b.set_title(L("(a) diode A: straight line of the datasheet", "(a) diodo A: retta del datasheet"), loc="left")
    d, c = exB["d"], exB["c"]
    i = np.linspace(0.2, 200.0, 800)
    b = ax[1]
    b.plot(i, exB["VT0"] + exB["rT"]*i, color=INK, lw=0.9, ls=(0, (4, 2)))
    b.plot(i, lm.forward_voltage(i, exB["IS"], exB["n"], c["VT"], exB["RS"]), color=C_NOM)
    b.plot([p[0] for p in d["VF"]], [p[1] for p in d["VF"]], "o", ms=3.5, color=INK)
    note(b, L("model", "modello") + ", $n$ = " + loc_num("%.2f" % exB["n"]), (150, float(lm.forward_voltage(150.0, exB["IS"], exB["n"], c["VT"], exB["RS"]))),
         (-10, 14), ha="right")
    note(b, L("line through the\ntwo upper points", "retta per i due\npunti superiori"), (12, exB["VT0"] + exB["rT"]*12), (20, -24))
    b.set_xlim(0, 200); b.set_ylim(0.5, 2.9)
    b.set_xlabel(L("forward current (A)", "forward current (A)")); b.set_ylabel(L("forward voltage (V)", "forward voltage (V)"))
    b.set_title(L("(b) diode B: three points of the curve", "(b) diodo B: tre punti della curva"), loc="left")
    save(fig, "fig_forward")


def fig_cell(exA, exB):
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.1), sharex="col", gridspec_kw=dict(hspace=0.14, wspace=0.27,
                                                                                    height_ratios=[1.25, 1]))
    for col, (ex, us, unit, lab) in enumerate(((exA, 1e6, "µs", "A"), (exB, 1e9, "ns", "B"))):
        ch, c = ex["chk"], ex["c"]
        s, m, r = ch["s"], ch["m"], ch["r"]
        t0 = m["t0"]
        lo, hi = t0 - 0.8*r["ta"], t0 + r["ta"] + 6.5*r["tau_rr"]
        sel = (s["t"] >= lo) & (s["t"] <= hi)
        x = (s["t"][sel] - t0)*us
        b = ax[0, col]
        b.axhline(0, color=AXIS, lw=0.6)
        b.plot(x, s["i"][sel], color=C_NOM)
        b.plot(x, s["i_dif"][sel], color=C_WEAK, lw=1.0)
        tc = np.linspace(lo, hi, 1500)
        ic, _, _ = lm.ramp_waveform(tc - (t0 - c["IF"]/m["a"]), m["a"], c["IF"], c["tau"], c["TM"])
        b.plot((tc - t0)*us, ic, color=INK, lw=0.8, ls=(0, (4, 2)))
        b.set_ylabel(L("diode current (A)", "corrente del diodo (A)"))
        b.set_title(f"({'ab'[col]}) " + L("diode", "diodo") + f" {lab}", loc="left")
        b.set_ylim(-1.18*m["Irr"], 0.75*m["Irr"])
        if col == 0:
            kx = int(np.argmin(np.abs(x - (r["ta"] + 2.2*r["tau_rr"])*us)))
            note(b, L("terminals", "terminali"), (x[kx], s["i"][sel][kx]), (18, -14))
            note(b, L("closed form\n(dashed)", "forma chiusa\n(tratteggiata)"), ((r["ta"] + 1.1*r["tau_rr"])*us, -m["Irr"]*math.exp(-1.1)), (26, -20))
        else:
            kx = int(np.argmin(np.abs(x - (r["ta"] + 0.25*r["tau_rr"])*us)))
            note(b, L("terminals", "terminali"), (x[kx], s["i"][sel][kx]), (16, -12))
            kx = int(np.argmin(np.abs(x - (r["ta"] + 0.9*r["tau_rr"])*us)))
            note(b, L("diffusion part", "parte di diffusione"), (x[kx], s["i_dif"][sel][kx]), (24, -16))
        b = ax[1, col]
        b.axhline(-ex["d"]["VR"], color=MUTED, lw=0.7, ls=(0, (4, 3)))
        b.plot(x, s["v"][sel], color=C_NOM)
        b.set_ylabel(L("diode voltage (V)", "tensione del diodo (V)"))
        b.set_xlabel(L("time from the current zero", "tempo dallo zero della corrente") + f" ({unit})")
        b.set_xlim(x[0], x[-1])
        b.annotate("$-V_\\mathrm{R}$", (x[-1], -ex["d"]["VR"]), xytext=(-2, 3), textcoords="offset points", fontsize=7,
                   ha="right", va="bottom", color=INK2)
    save(fig, "fig_cell")


def fig_curves(exA, exB):
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.3), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.30))
    cols = (RAMP[1], RAMP[3], RAMP[5])
    for col, (ex, lab, span) in enumerate(((exA, "A", (0.1, 20.0)), (exB, "B", (0.1, 10.0)))):
        d, c = ex["d"], ex["c"]
        a = c["a"]*np.logspace(math.log10(span[0]), math.log10(span[1]), 200)
        ends = [[], []]
        for f, cc in zip((0.5, 1.0, 2.0), cols):
            y = lm.ramp_peak_norm(c["IF"]*f/(a*c["tau"]), c["rec"]["rho"])
            irr = y*a*c["tau"]
            qrr = (0.5*y*y + c["rec"]["rho"]*y)*a*c["tau"]**2
            ax[0, col].plot(a/1e6, qrr*1e6, color=cc); ax[1, col].plot(a/1e6, irr, color=cc)
            ends[0].append(qrr[-1]*1e6); ends[1].append(irr[-1])
        ax[0, col].plot([c["a"]/1e6], [c["Qrr"]*1e6], "o", ms=3.5, color=C_WEAK, zorder=5)
        ax[1, col].plot([c["a"]/1e6], [c["Irr"]], "o", ms=3.5, color=C_WEAK, zorder=5)
        if "Q_EXP" in d:
            ad = np.array(d["Q_RANGE"])
            ax[0, col].plot(ad/1e6, d["QRR"]*(ad/c["a"])**d["Q_EXP"]*1e6, color=INK, lw=0.9, ls=(0, (4, 2)))
            ax[0, col].annotate(L("curve of the\ndatasheet", "curva del\ndatasheet"), (22.0, d["QRR"]*(22e6/c["a"])**d["Q_EXP"]*1e6), xytext=(4, -22),
                                textcoords="offset points", fontsize=7, ha="left", va="top",
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
        for row in (0, 1):
            ax[row, col].set_xscale("log"); ax[row, col].set_yscale("log")
            ax[row, col].set_xlim(a[0]/1e6, a[-1]/1e6*3.2)                   # room for the labels of the curves
            plain_log(ax[row, col])
            end_labels(ax[row, col], a[-1]/1e6, ends[row], [loc_num(f"{c['IF']*f:g}") + " A" for f in (0.5, 1.0, 2.0)])
        ax[0, col].set_title(f"({'ab'[col]}) " + L("diode", "diodo") + f" {lab}", loc="left")
        ax[1, col].set_xlabel(L("rate of fall of the current (A/µs)", "velocità di discesa della corrente (A/µs)"))
        ax[0, col].set_ylabel(L("recovered charge (µC)", "recovered charge (µC)"))
        ax[1, col].set_ylabel(L("peak reverse current (A)", "peak reverse current (A)"))
    save(fig, "fig_curves")


# ---------------------------------------------------------------- main
def main():
    global LANG
    t0 = time.time()
    line("Lauritzen-Ma model and datasheet data: worked examples")
    line()
    section_general()
    exA = section_a()
    exB = section_b()
    with open(os.path.join(HERE, "lm_conversion_results.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(LOG) + "\n")
    for LANG in ("en", "it"):              # figures/fig_*.pdf, then figures/fig_*_it.pdf
        fig_waveform()
    with open(os.path.join(HERE, "lm_conversion_values.tex"), "w", encoding="utf-8") as fo:
        fo.write("% generated by lm_conversion_calc.py: do not edit by hand\n")
        for k, v in MAC.items():
            fo.write(f"\\newcommand{{\\lm{k}}}{{{v}}}\n")
    note_tex = os.path.join(HERE, "lauritzen_ma_datasheet_conversion.tex")
    if os.path.exists(note_tex):                             # every macro of the note has to be written above
        with open(note_tex, encoding="utf-8") as fi:
            missing = sorted(set(re.findall(r"\\lm([A-Za-z]+)", fi.read())) - set(MAC))
        if missing:
            print("WARNING: macros used in the note and not written by this script: " + ", ".join(missing))
    for LANG in ("en", "it"):
        fig_map(exA, exB)
        fig_inversion(exA, exB)
        fig_forward(exA, exB)
        fig_cell(exA, exB)
        fig_curves(exA, exB)
    print(f"done in {time.time() - t0:.0f} s: {len(MAC)} macros, figures in {FIGDIR}")


if __name__ == "__main__":
    main()

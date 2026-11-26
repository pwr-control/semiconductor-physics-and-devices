"""Worked examples of body_diode_lauritzen_ma.tex: the body diode of a power MOSFET as a Lauritzen-Ma diode,
from the datasheet of the MOSFET to the parameters tau and T_M, with the output capacitance taken apart.

The script applies the relations of the note (module body_diode_lm.py, which imports lm_conversion.py from the
folder research/power_semiconductor_devices/diode/lauritzen_ma_datasheet_conversion) to three devices and writes
every number quoted in the note:
  1. the output charge Q_oss(VR) of each device and the recovery that the output capacitance alone produces in the
     datasheet test (the capacitive floor), against the datasheet figures;
  2. the main example, IPT015N10N5 (Infineon OptiMOS 5, 100 V): stored charge Q_rr - Q_oss(VR), conversion from
     the charge and the recovery time under the two driver models, bracket of the softness, the set obtained when
     Q_oss is not subtracted, maximum values, forward characteristic, plausibility of the time constants;
  3. simulated recovery test with the complete device (diffusion charge, C_oss(v), series resistance, driver with
     Miller plateau or with constant slope, clamp with stray inductance), refinement of the set on the simulated
     test, sensitivity to the capacitance at low voltage;
  4. the channel in parallel: injection with the channel on, dead time shorter than the lifetime, the set at the
     slopes of an application;
  5. the two other devices, IRFB4110 (two temperatures, peak current given) and STP30N10F7 (150 C).
The device data are typical values read from the datasheets quoted in the note (November 2026); the assumed
quantities are marked below. Outputs, next to the script: body_diode_results.txt (log), body_diode_values.tex
(macros \\bd... of the note), figures/*.pdf. Needs numpy, scipy, matplotlib. About one minute.
Run:  python body_diode_calc.py"""
import math, os, sys, time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import body_diode_lm as bd                 # noqa: E402  relations added by this note (imports lm_conversion)
import lm_conversion as lm                 # noqa: E402  two-charge model, conversion, forward fits (sibling note)
sys.path.insert(0, bd.LMDIR)
import lm_conversion_calc as lmc           # noqa: E402  style of the figures of the sibling note

FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- device data (typical values of the datasheets)
# Infineon IPT015N10N5, OptiMOS 5, 100 V, final data sheet rev. 2.4, 2023-05-04
IPT = dict(
    name="IPT015N10N5", VDS=100.0, RDSON=1.5e-3, ID=353.0,          # V, Ohm (max at VGS = 10 V), A
    T=25.0,                                                          # C: temperature of the recovery data
    IF=100.0, A=1e8, VR=50.0,                                        # A, A/s, V: IF, diF/dt, VR of the recovery test
    QRR=316e-9, QRR_MAX=632e-9, TRR=103e-9, TRR_MAX=206e-9,          # C, s: reverse recovery charge and time
    QOSS=213e-9, QOSS_MAX=284e-9,                                    # C: output charge, VDD = 50 V
    COSS=1.8e-9, CRSS=80e-12, CISS=12e-9,                            # F: capacitances at VDS = 50 V, 1 MHz
    VSD=0.85, VSD_MAX=1.2, IVSD=100.0,                               # V, A: diode forward voltage at IF = 100 A, 25 C
    IDSS=1e-6, IDSS_T=100e-6,                                        # A: assumed leakage at 100 V, 25 C and 125 C
    # ---- assumptions of the note (not in the datasheet) ----
    VJ=5.0,                      # V: junction potential of the power law of C_oss (low-voltage shape)
    VJ_ALT=1.0,                  # V: alternative, steeper law, for the sensitivity check
    N=2.0,                       # emission coefficient of the body diode (high injection), alternative 1
    RS=0.4e-3,                   # Ohm: series resistance of the diode (substrate, metal, package), assumed
    LS=10e-9, RSER=1.0,          # H, Ohm: stray inductance and damping of the clamp in the simulated test
    D=18e-4,                     # m2/s: ambipolar diffusion constant of silicon at 25 C, for the plausibility check
    W_EPI=(6e-6, 10e-6))         # m: plausible range of the drift region of a 100 V device [M]
# Infineon (International Rectifier) IRFB4110, HEXFET, 100 V, data sheet of 2014-04-28
IRF = dict(name="IRFB4110", VDS=100.0, RDSON=4.5e-3, ID=120.0, IF=75.0, A=1e8, VR=85.0,
           T=(25.0, 125.0), QRR=(94e-9, 140e-9), QRR_MAX=(140e-9, 210e-9), TRR=(50e-9, 60e-9), TRR_MAX=(75e-9, 90e-9),
           IRRM=3.5, COSS=670e-12, COSS_TR=950e-12, COSS_TR_V=80.0, COSS_ER=820e-12, VSD_MAX=1.3, IVSD=75.0,
           VJ=5.0)
# STMicroelectronics STP30N10F7, STripFET F7, 100 V, DocID028799 rev. 1, 2016-02-01
STP = dict(name="STP30N10F7", VDS=100.0, RDSON=24e-3, ID=32.0, IF=32.0, A=1e8, VR=80.0, T=150.0,
           QRR=47e-9, TRR=41e-9, IRRM=2.3, COSS=290e-12, VSD=1.1, IVSD=32.0,
           VJ=1.0, M_ASSUMED=0.5)          # no Q_oss and no curve read: power law with an assumed exponent
P_END = 0.02                               # charge of a simulated waveform integrated down to this fraction of the peak
V_APP = 48.0                               # V: bus voltage of the application example
A_APP = (0.5e9, 1e9, 2e9)                  # A/s: slopes of the application example
TD_APP = (20e-9, 50e-9, 100e-9)            # s: dead times of the application example

INK, INK2, MUTED, GRID, AXIS = lmc.INK, lmc.INK2, lmc.MUTED, lmc.GRID, lmc.AXIS
C_NOM, C_WEAK, C_3, C_4, RAMP = lmc.C_NOM, lmc.C_WEAK, lmc.C_3, lmc.C_4, lmc.RAMP
LOG, MAC = [], {}


def line(s=""):
    print(s)
    LOG.append(s)


def rnd(value, d=1):
    return str(Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def put(name, value, d=1):
    """macro \\bd<name> of the note (letters only)."""
    assert name.isalpha(), name
    MAC[name] = value if isinstance(value, str) else rnd(value, d)


def pct(x, d=1):
    return ("+" if x >= 0 else "-") + rnd(abs(x)*100, d)


def sci(x, d=1):
    e = int(math.floor(math.log10(abs(x))))
    return f"{x/10**e:.{d}f}\\times10^{{{e}}}"


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)


def note(ax, text, xy, xytext, ha="left", va="center", arrow=True, color=None):
    ax.annotate(text, xy, xytext=xytext, textcoords="offset points", ha=ha, va=va, fontsize=7, color=color or INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=2) if arrow else None)


# ---------------------------------------------------------------- 1. output charge and capacitive floor
def coss_law(d):
    """(C0, m, Vj, Qoss(VR)) of the power law of each device from what its datasheet gives."""
    Vj = d["VJ"]
    if "QOSS" in d:                                     # Q_oss and C_oss at the same voltage (IPT015N10N5)
        C0, m = bd.fit_coss_point_charge(d["COSS"], d["VR"], d["QOSS"], Vj)
    elif "COSS_TR" in d:                                # C_oss(tr) over 0..80 V gives the charge (IRFB4110)
        C0, m = bd.fit_coss_two_voltages(d["COSS"], 50.0, d["COSS_TR"]*d["COSS_TR_V"], d["COSS_TR_V"], Vj)
    else:                                               # one capacitance only: exponent assumed (STP30N10F7)
        m = d["M_ASSUMED"]
        C0 = d["COSS"]*(1 + 50.0/Vj)**m
    return C0, m, Vj, float(bd.qoss(d["VR"], C0, Vj, m))


def section_floor():
    line("1. Output charge and the recovery of the output capacitance alone")
    rows = {}
    for key, d in (("Ipt", IPT), ("Irf", IRF), ("Stp", STP)):
        C0, m, Vj, Q = coss_law(d)
        cap = bd.capacitive_only(d["A"], Q)
        d["LAW"] = (C0, m, Vj, Q)
        line(f"   {d['name']}: C_oss law C0 = {C0*1e9:.2f} nF, Vj = {Vj:g} V, m = {m:.3f}; C_oss(50 V) = "
             f"{float(bd.coss(50, C0, Vj, m))*1e12:.0f} pF; Q_oss({d['VR']:g} V) = {Q*1e9:.1f} nC; capacitive floor at "
             f"{d['A']/1e6:.0f} A/us: Irrm = {cap['Irrm']:.2f} A, t = {cap['trr']*1e9:.1f} ns, Q = {cap['Qrr']*1e9:.0f} nC")
        put(key + "Czero", C0*1e9, 1); put(key + "M", m, 2); put(key + "Vj", Vj, 0); put(key + "Qoss", Q*1e9, 0)
        put(key + "CapIrrm", cap["Irrm"], 1); put(key + "CapTrr", cap["trr"]*1e9, 0)
        put(key + "Vr", d["VR"], 0); put(key + "If", d["IF"], 0); put(key + "Slope", d["A"]/1e6, 0)
        put(key + "Coss", d["COSS"]*1e12, 0); put(key + "Vds", d["VDS"], 0); put(key + "Rdson", d["RDSON"]*1e3, 1)
        rows[key] = dict(Q=Q, cap=cap)
    # datasheet figures against the floor
    put("IptQrr", IPT["QRR"]*1e9, 0); put("IptQrrMax", IPT["QRR_MAX"]*1e9, 0); put("IptTrr", IPT["TRR"]*1e9, 0)
    put("IptTrrMax", IPT["TRR_MAX"]*1e9, 0); put("IptQossMax", IPT["QOSS_MAX"]*1e9, 0)
    put("IptQossPct", IPT["QOSS"]/IPT["QRR"]*100, 0); put("IptQossMaxPct", IPT["QOSS_MAX"]/IPT["QRR_MAX"]*100, 0)
    put("IptQdif", (IPT["QRR"] - IPT["QOSS"])*1e9, 0); put("IptQdifMax", (IPT["QRR_MAX"] - IPT["QOSS_MAX"])*1e9, 0)
    put("IptQdifPct", (1 - IPT["QOSS"]/IPT["QRR"])*100, 0)
    line(f"   IPT015N10N5: Qrr = {IPT['QRR']*1e9:.0f} nC of which Q_oss = {IPT['QOSS']*1e9:.0f} nC ({IPT['QOSS']/IPT['QRR']*100:.0f} %); "
         f"stored charge at most {(IPT['QRR'] - IPT['QOSS'])*1e9:.0f} nC; maximum values: {IPT['QRR_MAX']*1e9:.0f} - {IPT['QOSS_MAX']*1e9:.0f} = "
         f"{(IPT['QRR_MAX'] - IPT['QOSS_MAX'])*1e9:.0f} nC")
    Q = rows["Irf"]["Q"]
    for k, (T, Qrr, trr) in enumerate(zip(IRF["T"], IRF["QRR"], IRF["TRR"])):
        tag = "IrfLo" if k == 0 else "IrfHi"
        put(tag + "T", T, 0); put(tag + "Qrr", Qrr*1e9, 0); put(tag + "Trr", trr*1e9, 0); put(tag + "Qdif", (Qrr - Q)*1e9, 0)
        put(tag + "QossPct", Q/Qrr*100, 0); put(tag + "QrrMax", IRF["QRR_MAX"][k]*1e9, 0); put(tag + "TrrMax", IRF["TRR_MAX"][k]*1e9, 0)
        line(f"   IRFB4110 at {T:.0f} C: Qrr = {Qrr*1e9:.0f} nC, Q_oss(85 V) = {Q*1e9:.1f} nC ({Q/Qrr*100:.0f} %), stored at most {(Qrr - Q)*1e9:.1f} nC")
    put("IrfIrrm", IRF["IRRM"], 1); put("IrfTri", 0.5*IRF["IRRM"]**2/IRF["A"]*1e9, 0)
    put("IrfQdifRatio", (IRF["QRR"][1] - Q)/(IRF["QRR"][0] - Q), 1)
    put("IrfTauRatio", math.sqrt((IRF["QRR"][1] - Q)/(IRF["QRR"][0] - Q)), 1)
    line(f"   IRFB4110: terminal triangle Irrm^2/(2a) = {0.5*IRF['IRRM']**2/IRF['A']*1e9:.0f} nC against a stored charge of "
         f"{(IRF['QRR'][0] - Q)*1e9:.0f} nC at 25 C: the three typical values are not consistent with the capacitance; "
         f"stored charge 125 C over 25 C: {(IRF['QRR'][1] - Q)/(IRF['QRR'][0] - Q):.1f} (lifetime ratio about "
         f"{math.sqrt((IRF['QRR'][1] - Q)/(IRF['QRR'][0] - Q)):.1f} on a long ramp)")
    Q = rows["Stp"]["Q"]
    put("StpQrr", STP["QRR"]*1e9, 0); put("StpTrr", STP["TRR"]*1e9, 0); put("StpIrrm", STP["IRRM"], 1); put("StpT", STP["T"], 0)
    put("StpQdif", (STP["QRR"] - Q)*1e9, 0); put("StpQossPct", Q/STP["QRR"]*100, 0)
    put("StpTri", 0.5*STP["IRRM"]**2/STP["A"]*1e9, 0); put("StpMassumed", STP["M_ASSUMED"], 1)
    line(f"   STP30N10F7 at 150 C: Qrr = {STP['QRR']*1e9:.0f} nC, Q_oss(80 V) estimated {Q*1e9:.0f} nC ({Q/STP['QRR']*100:.0f} %), "
         f"triangle Irrm^2/(2a) = {0.5*STP['IRRM']**2/STP['A']*1e9:.0f} nC > stored charge {(STP['QRR'] - Q)*1e9:.0f} nC")
    return rows


# ---------------------------------------------------------------- 2. IPT015N10N5: conversion
def convert_ipt():
    line("")
    line("2. IPT015N10N5: from the datasheet to tau and T_M")
    d = IPT
    a, IF, VR, Qrr, trr = d["A"], d["IF"], d["VR"], d["QRR"], d["TRR"]
    C0, m, Vj, Qoss = d["LAW"]
    Qdif = Qrr - Qoss
    VT = lm.thermal_voltage(d["T"])
    out = dict(a=a, IF=IF, VR=VR, Qoss=Qoss, Qdif=Qdif, VT=VT)
    # --- plateau model from (Qrr, trr), the three conventions
    res = {}
    for conv in ("p25", "p10", "line"):
        Irr, trr_, info = bd.invert_terminal(a, Qoss, Qrr=Qrr, trr=trr, driver="plateau", convention=conv)
        res[conv] = (Irr, trr_, info)
    Irr, tau_rr, info = res["p25"]
    tau, TM, it = lm.lm_from_tail(a, IF, Irr, tau_rr)
    tauL, TML, _ = lm.lm_from_tail(a, IF, Irr, tau_rr, long_ramp=True)
    r = lm.ramp_recovery(a, IF, tau, TM)
    spread = max(abs(res[c][0]/Irr - 1) for c in res)
    line(f"   stored charge Q_dif = Qrr - Q_oss(VR) = {Qrr*1e9:.0f} - {Qoss*1e9:.0f} = {Qdif*1e9:.0f} nC")
    line(f"   plateau model with (Q_dif, trr = {trr*1e9:.0f} ns): Irr = {Irr:.2f} A, ta = {Irr/a*1e9:.1f} ns, tau_rr = {tau_rr*1e9:.2f} ns, "
         f"flat top Tc = {info['Tc']*1e9:.1f} ns, residual tail {info['i_res']:.3f} A; conventions 25 %, 10 %, line differ by {spread*100:.2f} %")
    line(f"   exact conversion ({it} iterations): tau = {tau*1e9:.1f} ns, TM = {TM*1e9:.1f} ns; long-ramp closed form tau = {tauL*1e9:.1f}, TM = {TML*1e9:.1f} ns; "
         f"T1/tau = {r['x']:.1f}, IF/(a tau) = {r['lam']:.1f}, softness s = {r['s']:.2f}, rho = {r['rho']:.3f}")
    out.update(Irr=Irr, tau_rr=tau_rr, tau=tau, TM=TM, info=info, rec=r)
    put("IptIrr", Irr, 2); put("IptTa", Irr/a*1e9, 1); put("IptTaurr", tau_rr*1e9, 1); put("IptTc", info["Tc"]*1e9, 0)
    put("IptTau", tau*1e9, 0); put("IptTauOne", tau*1e9, 1); put("IptTm", TM*1e9, 0); put("IptTmOne", TM*1e9, 1)
    put("IptTauL", tauL*1e9, 1); put("IptTmL", TML*1e9, 1); put("IptIter", str(it))
    put("IptX", r["x"], 0); put("IptLam", r["lam"], 0); put("IptS", r["s"], 2); put("IptRho", r["rho"], 2)
    put("IptQtri", 0.5*Irr**2/a*1e9, 0); put("IptQtail", Irr*tau_rr*1e9, 0); put("IptConvSpread", spread*100, 2)
    put("IptIres", info["i_res"]*1e3, 1); put("IptTzero", IF/a*1e6, 0)
    # --- ramp model has no solution: what it would need
    try:
        bd.invert_terminal(a, Qoss, Qrr=Qrr, trr=trr, driver="ramp", convention="p25")
        line("   ramp model: solution found (unexpected)")
    except ValueError as e:
        need = 0.5*(a*trr)**2/a
        line(f"   ramp model with (Q_dif, trr): no solution ({e}); it would need a terminal triangle of {need*1e9:.0f} nC "
             f"(Irrm = a trr = {a*trr:.1f} A) against Qrr = {Qrr*1e9:.0f} nC")
        put("IptRampIrrm", a*trr, 1); put("IptRampTri", need*1e9, 0)
    # --- bracket with an assumed softness
    line("   softness assumed instead of trr (charge alone):")
    line("     s     Irr [A]  tau_rr [ns]  tau [ns]  TM [ns]  | plateau: Irrm, trr25 | ramp: Irrm, trr25")
    brk = {}
    for s, tag in ((0.25, "Quarter"), (0.5, "Half"), (1.0, "One"), (2.0, "Two")):
        Irr_s = lm.irr_from_qrr_softness(a, Qdif, s)
        trr_s = s*Irr_s/(2*a)
        tau_s, TM_s, _ = lm.lm_from_tail(a, IF, Irr_s, trr_s)
        tp, tr = bd.terminal_plateau(a, Irr_s, trr_s, Qoss), bd.terminal_ramp(a, Irr_s, trr_s, Qoss)
        line(f"   {s:5.2f}  {Irr_s:7.2f}  {trr_s*1e9:10.1f}  {tau_s*1e9:8.1f}  {TM_s*1e9:7.1f}  | {tp['Irrm']:5.2f} A, {tp['trr_p25']*1e9:6.1f} ns | "
             f"{tr['Irrm']:5.2f} A, {tr['trr_p25']*1e9:6.1f} ns")
        put("IptS" + tag + "Irr", Irr_s, 2); put("IptS" + tag + "Tau", tau_s*1e9, 0); put("IptS" + tag + "Tm", TM_s*1e9, 0)
        put("IptS" + tag + "TrrP", tp["trr_p25"]*1e9, 0); put("IptS" + tag + "TrrR", tr["trr_p25"]*1e9, 0)
        put("IptS" + tag + "IrrmR", tr["Irrm"], 1)
        brk[s] = (Irr_s, trr_s, tau_s, TM_s)
    out["bracket"] = brk
    # --- the set obtained without subtracting Q_oss (as the fitting hints of mos_dyn.ssc do)
    k25, k10 = lm.trr_factor("point", p=0.25), lm.trr_factor("point", p=0.1)
    naive = {}
    for kap, tag in ((k25, "N"), (2.0, "NTri"), (k10, "NTen")):
        Irr_n = lm.irr_from_qrr_trr(a, Qrr, trr, kap)
        tau_n, TM_n, trr_n = lm.lm_from_irr_trr(a, IF, Irr_n, trr, kap)
        naive[tag] = (Irr_n, trr_n, tau_n, TM_n)
        line(f"   Q_oss not subtracted, kappa = {kap:.2f}: Irr = {Irr_n:.2f} A, tau_rr = {trr_n*1e9:.1f} ns, tau = {tau_n*1e9:.0f} ns, TM = {TM_n*1e9:.0f} ns")
        put("Ipt" + tag + "Irr", Irr_n, 1); put("Ipt" + tag + "Tau", tau_n*1e9, 0); put("Ipt" + tag + "Tm", TM_n*1e9, 0)
        put("Ipt" + tag + "Taurr", trr_n*1e9, 0)
    put("IptKtwentyfive", k25, 2)
    out["naive"] = naive["N"]
    put("IptNTauRatio", naive["N"][2]/tau, 1); put("IptNTmRatio", naive["N"][3]/TM, 1)
    put("IptNQrrDouble", (Qrr + Qoss)*1e9, 0); put("IptNQrrDoublePct", Qoss/Qrr*100, 0)
    # --- maximum values
    Qoss_max = IPT["QOSS_MAX"]
    C0x, mx = bd.fit_coss_point_charge(2.3e-9, VR, Qoss_max, Vj)
    Irr_x, trr_x, info_x = bd.invert_terminal(a, Qoss_max, Qrr=IPT["QRR_MAX"], trr=IPT["TRR_MAX"], driver="plateau")
    tau_x, TM_x, _ = lm.lm_from_tail(a, IF, Irr_x, trr_x)
    line(f"   maximum values (Qrr {IPT['QRR_MAX']*1e9:.0f} nC, trr {IPT['TRR_MAX']*1e9:.0f} ns, Q_oss {Qoss_max*1e9:.0f} nC): Irr = {Irr_x:.2f} A, "
         f"tau_rr = {trr_x*1e9:.1f} ns, tau = {tau_x*1e9:.0f} ns, TM = {TM_x*1e9:.0f} ns, softness {2*trr_x/(Irr_x/a):.2f}")
    put("IptXIrr", Irr_x, 1); put("IptXTaurr", trr_x*1e9, 0); put("IptXTau", tau_x*1e9, 0); put("IptXTm", TM_x*1e9, 0)
    put("IptXS", 2*trr_x/(Irr_x/a), 2); put("IptXTauRatio", tau_x/tau, 1)
    out["max"] = (Irr_x, trr_x, tau_x, TM_x)
    # --- plausibility: width of the base
    D = d["D"]
    Lw = math.sqrt(D*tau)
    W_par = 2*math.sqrt(3*D*TM)
    W_store = math.sqrt(2*D*tau)
    lo, hi = d["W_EPI"]
    t_lo, t_hi = lo**2/(2*D), hi**2/(2*D)
    line(f"   plausibility with D = {D*1e4:.0f} cm2/s: diffusion length sqrt(D tau) = {Lw*1e6:.0f} um; base width from TM (parabolic "
         f"profile) {W_par*1e6:.1f} um; a base of {lo*1e6:.0f}-{hi*1e6:.0f} um stores for W^2/(2D) = {t_lo*1e9:.0f}-{t_hi*1e9:.0f} ns "
         f"(tau found: {tau*1e9:.0f} ns); W from tau as W^2/(2D): {W_store*1e6:.0f} um")
    put("IptDcm", D*1e4, 0); put("IptLdiff", Lw*1e6, 0); put("IptWpar", W_par*1e6, 0); put("IptWstore", W_store*1e6, 0)
    put("IptWlo", lo*1e6, 0); put("IptWhi", hi*1e6, 0); put("IptTlo", t_lo*1e9, 0); put("IptThi", t_hi*1e9, 0)
    put("IptTmGeom", (0.5*8e-6)**2/(3*D)*1e9, 1)
    # --- forward characteristic: one point, n assumed, RS assumed
    for n, tag in ((2.0, "Two"), (1.0, "One")):
        IS = d["IVSD"]/math.expm1((d["VSD"] - d["RS"]*d["IVSD"])/(n*VT))
        ISstar = IS*(tau + TM)/tau
        v10 = float(lm.forward_voltage(10.0, IS, n, VT, d["RS"]))
        v300 = float(lm.forward_voltage(300.0, IS, n, VT, d["RS"]))
        line(f"   forward: n = {n:g}, RS = {d['RS']*1e3:.1f} mOhm assumed, VSD = {d['VSD']} V at {d['IVSD']:.0f} A: IS = {IS:.3g} A, "
             f"IS* = {ISstar:.3g} A; model gives {v10:.3f} V at 10 A and {v300:.3f} V at 300 A")
        put("IptIs" + tag, sci(IS, 1)); put("IptIsStar" + tag, sci(ISstar, 1)); put("IptVten" + tag, v10, 2); put("IptVthree" + tag, v300, 2)
        if n == d["N"]:
            out["IS"], out["n"] = IS, n
    put("IptVt", VT*1e3, 1); put("IptRs", d["RS"]*1e3, 1); put("IptVsd", d["VSD"], 2); put("IptVsdMax", d["VSD_MAX"], 1)
    put("IptIvsd", d["IVSD"], 0); put("IptN", d["N"], 0); put("IptIsFactor", (tau + TM)/tau, 2)
    put("IptRr", d["VDS"]/d["IDSS"]/1e6, 0); put("IptRrHot", d["VDS"]/d["IDSS_T"]/1e6, 0)
    # --- channel in parallel
    for I, tag in ((100.0, "Hundred"), (300.0, "Three")):
        i_d, v = bd.channel_injection(I, d["RDSON"], out["IS"], out["n"], VT, d["RS"])
        line(f"   channel on, {I:.0f} A: v = {v*1e3:.0f} mV, diode current {i_d:.3g} A, stored charge {tau*i_d:.3g} C")
        put("IptChV" + tag, v*1e3, 0); put("IptChI" + tag, sci(i_d, 1) if i_d > 0 else "0")
        put("IptChQ" + tag, sci(tau*i_d, 1) if i_d > 0 else "0")
    # --- exponents at the datasheet point
    mI, mQ = lm.ramp_exponents(a, IF, tau, TM)
    put("IptExpIa", mI, 2); put("IptExpQa", mQ, 2)
    line(f"   local exponents of the set at the datasheet point: Irr ~ a^{mI:.2f}, Qrr ~ a^{mQ:.2f}")
    return out


# ---------------------------------------------------------------- 3. simulated test
def device_dict(c, tau, TM, Vj=None, C0=None, m=None):
    d = IPT
    C0_, m_, Vj_, _ = d["LAW"]
    return dict(tau=tau, TM=TM, IS=c["IS"], n=c["n"], VT=c["VT"], RS=d["RS"], C0=C0 or C0_, Vj=Vj or Vj_, m=m or m_,
                Rr=d["VDS"]/d["IDSS"])


def run_test(dev, driver):
    s = bd.simulate_test(dev, IPT["IF"], IPT["VR"], IPT["A"], driver=driver, Ls=IPT["LS"], Rs=IPT["RSER"])
    return s, bd.measure_test(s, p_end=P_END)


def refine(c, driver, Vj, C0, m, targets):
    """(tau, TM) such that the simulated test returns the datasheet charge (to the current zero) and recovery
    time (25 % convention)."""
    Qt, tt = targets

    def resid(x):
        dev = device_dict(c, math.exp(x[0]), math.exp(x[1]), Vj, C0, m)
        _, mm = run_test(dev, driver)
        return [math.log(mm["Qrr_zero"]/Qt), math.log(mm["trr_25"]/tt)]
    x0 = [math.log(c["tau"]), math.log(c["TM"])]
    sol = least_squares(resid, x0, xtol=1e-6, ftol=1e-8, diff_step=1e-3, max_nfev=60)
    return math.exp(sol.x[0]), math.exp(sol.x[1]), sol


def section_sim(c):
    line("")
    line("3. Simulated recovery test (complete device)")
    d = IPT
    C0, m, Vj, Qoss = d["LAW"]
    tau, TM = c["tau"], c["TM"]
    sims = {}
    dev = device_dict(c, tau, TM)
    for drv in ("plateau", "ramp"):
        t0_ = time.time()
        s, mm = run_test(dev, drv)
        sims[drv] = (s, mm)
        tag = "P" if drv == "plateau" else "R"
        line(f"   closed-form set, {drv} driver ({time.time() - t0_:.1f} s): junction blocks at {(s['T1'] - s['t0'])*1e9:.1f} ns after the zero "
             f"(closed form {c['Irr']/c['a']*1e9:.1f} ns), voltage reaches VR at {(s['T2'] - s['t0'])*1e9:.1f} ns; terminals: Irrm = {mm['Irr']:.2f} A, "
             f"Qrr to the zero = {mm['Qrr_zero']*1e9:.0f} nC (to 2 %: {mm['Qrr']*1e9:.0f} nC), trr 25 % = {mm['trr_25']*1e9:.0f} ns, 10 % = {mm['trr_10']*1e9:.0f} ns, "
             f"line = {mm['trr_line']*1e9:.0f} ns; peak voltage {s['v'].max():.0f} V")
        put("IptSim" + tag + "Tone", (s["T1"] - s["t0"])*1e9, 0); put("IptSim" + tag + "Ttwo", (s["T2"] - s["t0"])*1e9, 0)
        put("IptSim" + tag + "Irrm", mm["Irr"], 1); put("IptSim" + tag + "Qrr", mm["Qrr_zero"]*1e9, 0)
        put("IptSim" + tag + "Trr", mm["trr_25"]*1e9, 0); put("IptSim" + tag + "TrrLine", mm["trr_line"]*1e9, 0)
        put("IptSim" + tag + "Vpk", s["v"].max(), 0); put("IptSim" + tag + "QrrErr", pct(mm["Qrr_zero"]/d["QRR"] - 1, 0))
        put("IptSim" + tag + "TrrErr", pct(mm["trr_25"]/d["TRR"] - 1, 0))
    # where the extra charge comes from: the capacitance from the forward voltage to zero, and the overshoot
    s, mm = sims["plateau"]
    vj0 = c["n"]*c["VT"]*math.log1p(d["IF"]/c["IS"])
    q_fwd = C0*vj0
    q_over = float(bd.qoss(s["v"].max(), C0, Vj, m)) - Qoss
    line(f"   extra charge of the simulation: C_oss from the forward voltage {vj0:.2f} V to zero about {q_fwd*1e9:.0f} nC, overshoot to "
         f"{s['v'].max():.0f} V about {q_over*1e9:.0f} nC")
    put("IptSimQfwd", q_fwd*1e9, 0); put("IptSimQover", q_over*1e9, 0); put("IptVjzero", vj0, 2)
    # refinement on the simulated test
    t0_ = time.time()
    tau_r, TM_r, sol = refine(c, "plateau", Vj, C0, m, (d["QRR"], d["TRR"]))
    dev_r = device_dict(c, tau_r, TM_r)
    s_r, mm_r = run_test(dev_r, "plateau")
    line(f"   refined on the simulated test, plateau driver ({time.time() - t0_:.0f} s, {sol.nfev} simulations): tau = {tau_r*1e9:.1f} ns "
         f"({(tau_r/tau - 1)*100:+.0f} %), TM = {TM_r*1e9:.1f} ns ({(TM_r/TM - 1)*100:+.0f} %); check Qrr = {mm_r['Qrr_zero']*1e9:.0f} nC, "
         f"trr = {mm_r['trr_25']*1e9:.0f} ns, Irrm = {mm_r['Irr']:.2f} A, s = {2*lm.tail_time(tau_r, TM_r)/(mm_r['Irr']/d['A']):.2f}")
    put("IptRefTau", tau_r*1e9, 0); put("IptRefTm", TM_r*1e9, 0); put("IptRefTauErr", pct(tau_r/tau - 1, 0)); put("IptRefTmErr", pct(TM_r/TM - 1, 0))
    put("IptRefQrr", mm_r["Qrr_zero"]*1e9, 0); put("IptRefTrr", mm_r["trr_25"]*1e9, 0); put("IptRefIrrm", mm_r["Irr"], 1)
    put("IptRefTaurr", lm.tail_time(tau_r, TM_r)*1e9, 1); put("IptRefNsim", str(sol.nfev))
    put("IptRefS", 2*lm.tail_time(tau_r, TM_r)/(mm_r["Irr"]/d["A"]), 2)
    put("IptXScale", math.sqrt((d["QRR_MAX"] - d["QOSS_MAX"])/(d["QRR"] - d["QOSS"])), 1)
    sims["refined"] = (s_r, mm_r)
    # the same with the steeper capacitance law (sensitivity to C_oss at low voltage)
    C0a, ma = bd.fit_coss_point_charge(d["COSS"], d["VR"], d["QOSS"], d["VJ_ALT"])
    t0_ = time.time()
    tau_a, TM_a, sol_a = refine(c, "plateau", d["VJ_ALT"], C0a, ma, (d["QRR"], d["TRR"]))
    line(f"   refined with the steeper law (Vj = {d['VJ_ALT']:g} V, C0 = {C0a*1e9:.0f} nF, m = {ma:.2f}; {time.time() - t0_:.0f} s): "
         f"tau = {tau_a*1e9:.1f} ns, TM = {TM_a*1e9:.1f} ns")
    put("IptAltCzero", C0a*1e9, 0); put("IptAltM", ma, 2); put("IptAltTau", tau_a*1e9, 0); put("IptAltTm", TM_a*1e9, 0)
    put("IptAltTauErr", pct(tau_a/tau_r - 1, 0)); put("IptAltTmErr", pct(TM_a/TM_r - 1, 0))
    # ramp driver with the refined set: what another test circuit would have printed
    s_rr, mm_rr = run_test(dev_r, "ramp")
    line(f"   refined set on the ramp driver: Irrm = {mm_rr['Irr']:.2f} A, Qrr = {mm_rr['Qrr_zero']*1e9:.0f} nC, trr 25 % = {mm_rr['trr_25']*1e9:.0f} ns")
    put("IptRefRIrrm", mm_rr["Irr"], 1); put("IptRefRQrr", mm_rr["Qrr_zero"]*1e9, 0); put("IptRefRTrr", mm_rr["trr_25"]*1e9, 0)
    sims["refined_ramp"] = (s_rr, mm_rr)
    # the set without the subtraction of Q_oss, in the same simulation: double counting
    Irr_n, trr_n, tau_n, TM_n = c["naive"]
    dev_n = device_dict(c, tau_n, TM_n)
    s_n, mm_n = run_test(dev_n, "plateau")
    line(f"   set converted without subtracting Q_oss (tau = {tau_n*1e9:.0f} ns, TM = {TM_n*1e9:.0f} ns) with C_oss in parallel: Irrm = {mm_n['Irr']:.2f} A, "
         f"Qrr = {mm_n['Qrr_zero']*1e9:.0f} nC ({(mm_n['Qrr_zero']/d['QRR'] - 1)*100:+.0f} %), trr = {mm_n['trr_25']*1e9:.0f} ns")
    put("IptNSimQrr", mm_n["Qrr_zero"]*1e9, 0); put("IptNSimQrrErr", pct(mm_n["Qrr_zero"]/d["QRR"] - 1, 0))
    put("IptNSimTrr", mm_n["trr_25"]*1e9, 0); put("IptNSimIrrm", mm_n["Irr"], 1)
    sims["naive"] = (s_n, mm_n)
    # the capacitance alone (a very short lifetime)
    dev_c = device_dict(c, 0.5e-9, 0.1e-9)
    s_c, mm_c = run_test(dev_c, "ramp")
    cap = bd.capacitive_only(d["A"], Qoss)
    line(f"   capacitance alone (tau = 0.5 ns), ramp driver: Irrm = {mm_c['Irr']:.2f} A (closed form {cap['Irrm']:.2f}), Qrr = {mm_c['Qrr_zero']*1e9:.0f} nC, "
         f"trr 25 % = {mm_c['trr_25']*1e9:.0f} ns (closed form {cap['trr']*1e9:.0f} ns)")
    put("IptCapSimIrrm", mm_c["Irr"], 1); put("IptCapSimQrr", mm_c["Qrr_zero"]*1e9, 0); put("IptCapSimTrr", mm_c["trr_25"]*1e9, 0)
    sims["cap"] = (s_c, mm_c)
    put("IptLs", d["LS"]*1e9, 0); put("IptRser", d["RSER"], 1)
    c["refined"] = (tau_r, TM_r)
    return sims


# ---------------------------------------------------------------- 4. channel, dead time and slope
def section_app(c):
    line("")
    line("4. Dead time and the slopes of an application")
    tau, TM = c["refined"]
    IF, a_ds = IPT["IF"], IPT["A"]
    rho = lm.tail_time(tau, TM)/tau
    r_ds = lm.ramp_recovery(a_ds, IF, tau, TM)
    line(f"   set tau = {tau*1e9:.1f} ns, TM = {TM*1e9:.1f} ns (rho = {rho:.2f}); datasheet slope: IF/(a tau) = {r_ds['lam']:.0f}, Q_dif = {r_ds['Qrr']*1e9:.0f} nC")
    put("AppTau", tau*1e9, 0); put("AppTm", TM*1e9, 0); put("AppRho", rho, 2); put("AppLamDs", r_ds["lam"], 0)
    # dead time at the datasheet slope: no effect
    for td in (20e-9, 50e-9):
        r = bd.ramp_recovery_dead_time(a_ds, IF, tau, TM, td)
        line(f"   datasheet slope, dead time {td*1e9:.0f} ns: Q_dif = {r['Qrr']*1e9:.1f} nC ({(r['Qrr']/r_ds['Qrr'] - 1)*100:+.2f} %)")
    put("AppQdsTdTwenty", bd.ramp_recovery_dead_time(a_ds, IF, tau, TM, 20e-9)["Qrr"]*1e9, 1)
    # application slopes
    C0_, m_, Vj_, _ = IPT["LAW"]
    Qoss_app = float(bd.qoss(V_APP, C0_, Vj_, m_))
    put("AppVbus", V_APP, 0); put("AppQoss", Qoss_app*1e9, 0)
    put("AppDsIrr", r_ds["Irr"], 1); put("AppDsQ", r_ds["Qrr"]*1e9, 0); put("AppDsQoverQoss", r_ds["Qrr"]/Qoss_app, 1)
    put("AppDsSlope", a_ds/1e9, 1)
    line(f"   application: {V_APP:.0f} V bus, Q_oss({V_APP:.0f} V) = {Qoss_app*1e9:.0f} nC")
    line("     slope [A/ns]  lam   | steady: Irr [A], Q_dif [nC], Q_dif/Q_oss | td = 20 ns | td = 50 ns | td = 100 ns  (Irr, Q_dif)")
    for a_app, tag in zip(A_APP, ("Half", "One", "Two")):
        r_inf = lm.ramp_recovery(a_app, IF, tau, TM)
        cells = []
        for td, ttag in zip(TD_APP, ("Twenty", "Fifty", "Hundred")):
            r = bd.ramp_recovery_dead_time(a_app, IF, tau, TM, td)
            cells.append(f"{r['Irr']:.1f} A, {r['Qrr']*1e9:.0f} nC")
            put("App" + tag + ttag + "Irr", r["Irr"], 1); put("App" + tag + ttag + "Q", r["Qrr"]*1e9, 0)
            put("App" + tag + ttag + "QPct", r["Qrr"]/r_inf["Qrr"]*100, 0)
        line(f"     {a_app/1e9:6.1f}      {r_inf['lam']:5.2f} | {r_inf['Irr']:6.1f}, {r_inf['Qrr']*1e9:6.0f}, {r_inf['Qrr']/Qoss_app:4.1f} | " + " | ".join(cells))
        put("App" + tag + "Lam", r_inf["lam"], 2); put("App" + tag + "Irr", r_inf["Irr"], 1); put("App" + tag + "Q", r_inf["Qrr"]*1e9, 0)
        put("App" + tag + "QRatio", r_inf["Qrr"]/r_ds["Qrr"], 1); put("App" + tag + "QoverQoss", r_inf["Qrr"]/Qoss_app, 1)
        put("App" + tag + "Slope", a_app/1e9, 1); put("App" + tag + "X", r_inf["x"], 2)
    # the abrupt limit
    line(f"   abrupt-commutation limit of the set: Irr = IF tau/TM = {IF*tau/TM:.0f} A, Q = IF (tau - tau_rr) = {IF*(tau - lm.tail_time(tau, TM))*1e9:.0f} nC")
    put("AppIrrAbrupt", IF*tau/TM, 0); put("AppQAbrupt", IF*(tau - lm.tail_time(tau, TM))*1e9, 0)
    put("AppTdTwenty", 20, 0); put("AppTdFifty", 50, 0); put("AppTdHundred", 100, 0)
    # charge build-up after the channel turns off
    for td, tag in ((20e-9, "Twenty"), (50e-9, "Fifty"), (100e-9, "Hundred")):
        put("AppFill" + tag, (1 - math.exp(-td/tau))*100, 0)
    line(f"   charge in the base after the channel turns off, fraction of tau IF: 20 ns {100*(1 - math.exp(-20e-9/tau)):.0f} %, "
         f"50 ns {100*(1 - math.exp(-50e-9/tau)):.0f} %, 100 ns {100*(1 - math.exp(-100e-9/tau)):.0f} %")


# ---------------------------------------------------------------- 5. the other two devices
def section_others():
    line("")
    line("5. IRFB4110 and STP30N10F7")
    C0, m, Vj, Q = IRF["LAW"]
    a = IRF["A"]
    # 25 C: no set under either model
    for drv in ("plateau", "ramp"):
        for data in (dict(Qrr=IRF["QRR"][0], Irrm=IRF["IRRM"]), dict(Qrr=IRF["QRR"][0], trr=IRF["TRR"][0])):
            try:
                bd.invert_terminal(a, Q, driver=drv, **data)
                line(f"   IRFB4110 25 C, {drv}, {list(data)}: solution found")
            except ValueError as e:
                line(f"   IRFB4110 25 C, {drv}, {list(data)}: no solution ({e})")
    # with the capacitive floor
    cap = bd.capacitive_only(a, Q)
    line(f"   IRFB4110 25 C: datasheet Irrm {IRF['IRRM']:.1f} A, trr {IRF['TRR'][0]*1e9:.0f} ns, Qrr {IRF['QRR'][0]*1e9:.0f} nC; capacitance alone: "
         f"{cap['Irrm']:.1f} A, {cap['trr']*1e9:.0f} ns, {cap['Qrr']*1e9:.0f} nC")
    # 125 C: plateau model from (Qrr, trr)
    Irr, trr_, info = bd.invert_terminal(a, Q, Qrr=IRF["QRR"][1], trr=IRF["TRR"][1], driver="plateau")
    tau, TM, _ = lm.lm_from_tail(a, IRF["IF"], Irr, trr_)
    line(f"   IRFB4110 125 C, plateau model from (Qrr, trr): Irr = {Irr:.2f} A, tau_rr = {trr_*1e9:.1f} ns, tau = {tau*1e9:.0f} ns, TM = {TM*1e9:.1f} ns, "
         f"softness {2*trr_/(Irr/a):.2f}; the datasheet gives no Irrm at 125 C")
    put("IrfHiIrr", Irr, 1); put("IrfHiTaurr", trr_*1e9, 1); put("IrfHiTau", tau*1e9, 0); put("IrfHiTm", TM*1e9, 1); put("IrfHiS", 2*trr_/(Irr/a), 2)
    # what a stored charge of 15 nC means on a long ramp: tau from Q = a (tau^2 - tau_rr^2)/2 with the softness of 125 C
    Qd = IRF["QRR"][0] - Q
    rho = lm.tail_time(tau, TM)/tau
    tau25 = math.sqrt(2*Qd/(a*(1 - rho**2)))
    line(f"   IRFB4110 25 C: a stored charge of {Qd*1e9:.0f} nC with the softness of 125 C corresponds to tau = {tau25*1e9:.0f} ns, "
         f"Irr = {a*tau25*(1 - rho):.2f} A (capacitance alone {cap['Irrm']:.1f} A)")
    put("IrfLoTau", tau25*1e9, 0); put("IrfLoIrr", a*tau25*(1 - rho), 1)
    put("IrfCossTr", IRF["COSS_TR"]*1e12, 0); put("IrfCossEr", IRF["COSS_ER"]*1e12, 0); put("IrfCossTrV", IRF["COSS_TR_V"], 0)
    put("IrfQossEighty", IRF["COSS_TR"]*IRF["COSS_TR_V"]*1e9, 0)
    # STP30N10F7
    C0, m, Vj, Q = STP["LAW"]
    a = STP["A"]
    cap = bd.capacitive_only(a, Q)
    line(f"   STP30N10F7 150 C: datasheet Irrm {STP['IRRM']:.1f} A, trr {STP['TRR']*1e9:.0f} ns, Qrr {STP['QRR']*1e9:.0f} nC; capacitance alone (estimate): "
         f"{cap['Irrm']:.1f} A, {cap['trr']*1e9:.0f} ns, {cap['Qrr']*1e9:.0f} nC")
    for drv in ("plateau", "ramp"):
        try:
            Irr, trr_, info = bd.invert_terminal(a, Q, Qrr=STP["QRR"], trr=STP["TRR"], driver=drv)
            tau, TM, _ = lm.lm_from_tail(a, STP["IF"], Irr, trr_)
            line(f"   STP30N10F7, {drv} from (Qrr, trr): Irr = {Irr:.2f} A, tau_rr = {trr_*1e9:.1f} ns, tau = {tau*1e9:.0f} ns, TM = {TM*1e9:.1f} ns")
            if drv == "plateau":
                put("StpIrr", Irr, 1); put("StpTaurr", trr_*1e9, 1); put("StpTau", tau*1e9, 0); put("StpTm", TM*1e9, 1)
                put("StpS", 2*trr_/(Irr/STP["A"]), 2)
        except ValueError as e:
            line(f"   STP30N10F7, {drv} from (Qrr, trr): no solution ({e})")
            if drv == "plateau":
                put("StpIrr", "--"); put("StpTau", "--"); put("StpTm", "--")
    # Q_rr * R_DS(on) of the three (die-area scaling)
    for key, d, Qrr in (("Ipt", IPT, IPT["QRR"]), ("Irf", IRF, IRF["QRR"][0]), ("Stp", STP, STP["QRR"])):
        put(key + "QrrR", Qrr*d["RDSON"]*1e12, 2); put(key + "QossR", d["LAW"][3]*d["RDSON"]*1e12, 2)
        line(f"   {d['name']}: Qrr*Rds(on) = {Qrr*d['RDSON']*1e12:.2f} nC mOhm, Q_oss(VR)*Rds(on) = {d['LAW'][3]*d['RDSON']*1e12:.2f} nC mOhm")


# ---------------------------------------------------------------- figures
def fig_floor(rows):
    """The datasheet charge of the three devices and the part of it that is the output charge."""
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.32))
    items = [("IPT015N10N5\ntyp., 25 °C", IPT["QRR"], IPT["QOSS"]), ("IPT015N10N5\nmax., 25 °C", IPT["QRR_MAX"], IPT["QOSS_MAX"]),
             ("IRFB4110\ntyp., 25 °C", IRF["QRR"][0], IRF["LAW"][3]), ("IRFB4110\ntyp., 125 °C", IRF["QRR"][1], IRF["LAW"][3]),
             ("STP30N10F7\ntyp., 150 °C", STP["QRR"], STP["LAW"][3])]
    x = np.arange(len(items))*1.25
    b = ax[0]
    b.bar(x, [q*1e9 for _, _, q in items], width=0.6, color=C_WEAK, label="$Q_\\mathrm{oss}(V_\\mathrm{R})$")
    b.bar(x, [(Qrr - q)*1e9 for _, Qrr, q in items], bottom=[q*1e9 for _, _, q in items], width=0.6, color=C_NOM,
          label="$Q_\\mathrm{rr}-Q_\\mathrm{oss}$: stored charge")
    for k, (lab, Qrr, q) in enumerate(items):
        b.annotate(f"{q/Qrr*100:.0f} %", (k, q*1e9/2), ha="center", va="center", fontsize=7, color="white")
    b.set_xticks(x); b.set_xticklabels([it[0] for it in items], fontsize=6)
    b.set_ylabel("charge (nC)")
    b.set_title("(a) datasheet $Q_\\mathrm{rr}$ and the output charge", loc="left")
    b.legend(loc="upper right", fontsize=7)
    b.grid(axis="x", visible=False)
    # (b) capacitive floor against the datasheet peak and time
    b = ax[1]
    devs = [("IPT015N10N5", rows["Ipt"]["cap"], None, IPT["TRR"]), ("IRFB4110", rows["Irf"]["cap"], IRF["IRRM"], IRF["TRR"][0]),
            ("STP30N10F7", rows["Stp"]["cap"], STP["IRRM"], STP["TRR"])]
    for k, (name, cap, Irrm, trr) in enumerate(devs):
        b.plot(cap["trr"]*1e9, cap["Irrm"], "o", color=C_WEAK, ms=5, mfc="white", mew=1.2)
        if Irrm is not None:
            b.plot(trr*1e9, Irrm, "s", color=C_NOM, ms=4.5)
            b.annotate("", (trr*1e9, Irrm), (cap["trr"]*1e9, cap["Irrm"]), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
        else:
            b.axvline(trr*1e9, color=C_NOM, lw=0.8, ls=(0, (3, 2)))
        b.annotate(name, (cap["trr"]*1e9, cap["Irrm"]), xytext=(5, 6 if k != 2 else -9), textcoords="offset points", fontsize=6.5)
    b.set_xlabel("$t_\\mathrm{rr}$ (ns)")
    b.set_ylabel("$I_\\mathrm{rrm}$ (A)")
    b.set_xlim(0, 120); b.set_ylim(0, 8)
    b.set_title("(b) capacitance alone (circles) and datasheet (squares)", loc="left", fontsize=7.5)
    b.annotate("datasheet $t_\\mathrm{rr}$ of\nIPT015N10N5\n(no $I_\\mathrm{rrm}$ given)", (IPT["TRR"]*1e9, 1.2), xytext=(-4, 0), textcoords="offset points",
               ha="right", va="center", fontsize=6.5, color=INK2)
    save(fig, "fig_floor")


def fig_coss(c):
    C0, m, Vj, Qoss = IPT["LAW"]
    C0a, ma = bd.fit_coss_point_charge(IPT["COSS"], IPT["VR"], IPT["QOSS"], IPT["VJ_ALT"])
    v = np.linspace(0, 100, 600)
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw=dict(wspace=0.32))
    b = ax[0]
    b.plot(v, bd.coss(v, C0, Vj, m)*1e9, color=C_NOM, label=f"$V_\\mathrm{{j}} = {Vj:g}$ V, $m = {m:.2f}$")
    b.plot(v, bd.coss(v, C0a, IPT["VJ_ALT"], ma)*1e9, color=C_WEAK, ls=(0, (4, 2)), label=f"$V_\\mathrm{{j}} = {IPT['VJ_ALT']:g}$ V, $m = {ma:.2f}$")
    b.plot([IPT["VR"]], [IPT["COSS"]*1e9], "s", color=INK, ms=4.5)
    note(b, "$C_\\mathrm{oss}$ of the table\n(50 V, 1 MHz)", (IPT["VR"], IPT["COSS"]*1e9), (10, 14))
    b.set_yscale("log"); b.set_ylim(0.8, 40)
    b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: f"{y:g}"))
    b.set_xlabel("$V_\\mathrm{DS}$ (V)"); b.set_ylabel("$C_\\mathrm{oss}$ (nF)")
    b.set_title("(a) power law through the two data of the table", loc="left")
    b.legend(loc="upper right", fontsize=7)
    b = ax[1]
    b.plot(v, bd.qoss(v, C0, Vj, m)*1e9, color=C_NOM)
    b.plot(v, bd.qoss(v, C0a, IPT["VJ_ALT"], ma)*1e9, color=C_WEAK, ls=(0, (4, 2)))
    b.plot([IPT["VR"]], [IPT["QOSS"]*1e9], "s", color=INK, ms=4.5)
    b.axhline(IPT["QRR"]*1e9, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    b.annotate("$Q_\\mathrm{rr}$ of the datasheet", (2, IPT["QRR"]*1e9), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    b.annotate("$Q_\\mathrm{rr}-Q_\\mathrm{oss}(V_\\mathrm{R})$:\nstored charge", (IPT["VR"], (IPT["QOSS"] + c["Qdif"]/2)*1e9), xytext=(8, 0),
               textcoords="offset points", fontsize=7, va="center")
    b.annotate("", (IPT["VR"], IPT["QRR"]*1e9), (IPT["VR"], IPT["QOSS"]*1e9), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    note(b, "$Q_\\mathrm{oss}$ of the table", (IPT["VR"], IPT["QOSS"]*1e9), (8, -16))
    b.set_xlabel("$V_\\mathrm{DS}$ (V)"); b.set_ylabel("$Q_\\mathrm{oss}$ (nC)")
    b.set_ylim(0, 380)
    b.set_title("(b) output charge and the datasheet $Q_\\mathrm{rr}$", loc="left")
    save(fig, "fig_coss")


def fig_test(c, sims):
    d = IPT
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.3), sharex="col", gridspec_kw=dict(hspace=0.12, wspace=0.26, height_ratios=[1.3, 1]))
    for col, (key, title) in enumerate((("refined", "(a) driver with Miller plateau"), ("refined_ramp", "(b) driver that keeps the slope"))):
        s, mm = sims[key]
        t0 = s["t0"]
        x = (s["t"] - t0)*1e9
        sel = (x > -25) & (x < 170)
        b = ax[0, col]
        b.axhline(0, color=AXIS, lw=0.6)
        b.plot(x[sel], s["i"][sel], color=C_NOM, label="terminals")
        b.plot(x[sel], s["i_dif"][sel], color=C_WEAK, lw=1.0, label="diffusion")
        b.plot(x[sel], s["i_c"][sel], color=C_3, lw=0.9, ls=(0, (3, 2)), label="$C_\\mathrm{oss}$")
        if col == 0:
            sn, mn = sims["naive"]
            xn = (sn["t"] - sn["t0"])*1e9
            seln = (xn > -25) & (xn < 170)
            b.plot(xn[seln], sn["i"][seln], color=INK, lw=0.8, ls=(0, (1.5, 1.5)), label="$Q_\\mathrm{oss}$ not subtracted")
            b.legend(loc="lower right", fontsize=6.5, ncol=2)
        b.axvline(d["TRR"]*1e9, color=MUTED, lw=0.7, ls=(0, (4, 3)))
        b.annotate("$t_\\mathrm{rr}$ of the\ndatasheet", (d["TRR"]*1e9, 2.0), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=INK2)
        b.set_ylim(-11.5, 4.5)
        b.set_ylabel("current (A)")
        b.set_title(title, loc="left")
        b = ax[1, col]
        b.axhline(d["VR"], color=MUTED, lw=0.7, ls=(0, (4, 3)))
        b.plot(x[sel], s["v"][sel], color=C_NOM)
        b.set_ylabel("reverse voltage (V)")
        b.set_xlabel("time from the current zero (ns)")
        b.set_xlim(-25, 170)
        b.annotate("$V_\\mathrm{R}$", (168, d["VR"]), xytext=(-2, 3), textcoords="offset points", fontsize=7, ha="right", color=INK2)
    save(fig, "fig_test")


def fig_bracket(c):
    a, IF, Qdif, Qoss, trr = c["a"], c["IF"], c["Qdif"], c["Qoss"], IPT["TRR"]
    ss = np.linspace(0.15, 3.0, 120)
    taus, TMs, trrP, trrR, IrrP, IrrR = [], [], [], [], [], []
    for s in ss:
        Irr = lm.irr_from_qrr_softness(a, Qdif, s)
        tr = s*Irr/(2*a)
        tau, TM, _ = lm.lm_from_tail(a, IF, Irr, tr)
        tp, trm = bd.terminal_plateau(a, Irr, tr, Qoss), bd.terminal_ramp(a, Irr, tr, Qoss)
        taus.append(tau*1e9); TMs.append(TM*1e9); trrP.append(tp["trr_p25"]*1e9); trrR.append(trm["trr_p25"]*1e9)
        IrrP.append(tp["Irrm"]); IrrR.append(trm["Irrm"])
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.5), gridspec_kw=dict(wspace=0.3))
    b = ax[0]
    b.plot(ss, taus, color=C_NOM, label="$\\tau$")
    b.plot(ss, TMs, color=C_WEAK, label="$T_\\mathrm{M}$")
    b.axvline(c["rec"]["s"], color=MUTED, lw=0.7, ls=(0, (4, 3)))
    b.annotate("softness that returns\nthe datasheet $t_\\mathrm{rr}$", (c["rec"]["s"], 95), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=INK2)
    b.set_xlabel("assumed softness $s = 2\\tau_\\mathrm{rr}/t_\\mathrm{a}$"); b.set_ylabel("time constant (ns)")
    b.set_ylim(0, 110); b.set_xlim(0, 3)
    b.set_title("(a) the two parameters from the stored charge alone", loc="left")
    b.legend(loc="lower right", fontsize=7)
    b = ax[1]
    b.plot(ss, trrP, color=C_NOM, label="plateau driver")
    b.plot(ss, trrR, color=C_WEAK, label="driver that keeps the slope")
    b.axhline(trr*1e9, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    b.annotate("datasheet $t_\\mathrm{rr}$", (0.08, trr*1e9), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    b.set_xlabel("assumed softness $s$"); b.set_ylabel("$t_\\mathrm{rr}$ at the terminals (ns)")
    b.set_xlim(0, 3); b.set_ylim(40, 170)
    b.set_title("(b) recovery time the test would show", loc="left")
    b.legend(loc="upper left", fontsize=7)
    save(fig, "fig_bracket")


def fig_deadtime(c):
    tau, TM = c["refined"]
    IF = IPT["IF"]
    rho = lm.tail_time(tau, TM)/tau
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.3))
    b = ax[0]
    tds = np.linspace(0, 4, 200)
    for k, lam in enumerate((0.3, 1.0, 3.0, 10.0)):
        a = IF/(lam*tau)
        r_inf = lm.ramp_recovery(a, IF, tau, TM)
        q = [bd.ramp_recovery_dead_time(a, IF, tau, TM, td*tau)["Qrr"]/r_inf["Qrr"] for td in tds]
        b.plot(tds, q, color=RAMP[1 + k], label=f"$I_\\mathrm{{F}}/(a\\tau) = {lam:g}$")
    b.set_xlabel("dead time $t_\\mathrm{d}/\\tau$"); b.set_ylabel("$Q_\\mathrm{dif}(t_\\mathrm{d})\\,/\\,Q_\\mathrm{dif}(\\infty)$")
    b.set_ylim(0, 1.05); b.set_xlim(0, 4)
    b.set_title(f"(a) stored charge against the dead time ($\\rho = {rho:.2f}$)", loc="left")
    b.legend(loc="lower right", fontsize=7)
    b = ax[1]
    aa = np.logspace(np.log10(0.05e9), np.log10(5e9), 150)
    C0_, m_, Vj_, _ = IPT["LAW"]
    Qoss_app = float(bd.qoss(V_APP, C0_, Vj_, m_))
    q_inf = [lm.ramp_recovery(a, IF, tau, TM)["Qrr"]*1e9 for a in aa]
    b.plot(aa/1e9, q_inf, color=C_NOM, label="steady state")
    for k, td in enumerate(TD_APP):
        q = [bd.ramp_recovery_dead_time(a, IF, tau, TM, td)["Qrr"]*1e9 for a in aa]
        b.plot(aa/1e9, q, color=RAMP[4 - k], lw=1.0, ls=(0, (4, 2)), label=f"$t_\\mathrm{{d}} = {td*1e9:.0f}$ ns")
    b.axhline(Qoss_app*1e9, color=C_WEAK, lw=0.9)
    b.annotate(f"$Q_\\mathrm{{oss}}({V_APP:.0f}$ V$)$", (0.055, Qoss_app*1e9), xytext=(0, 3), textcoords="offset points", fontsize=7, color=C_WEAK)
    b.plot([IPT["A"]/1e9], [lm.ramp_recovery(IPT["A"], IF, tau, TM)["Qrr"]*1e9], "s", color=INK, ms=4.5)
    note(b, "datasheet point", (IPT["A"]/1e9, lm.ramp_recovery(IPT["A"], IF, tau, TM)["Qrr"]*1e9), (8, -10))
    b.set_xscale("log"); b.set_yscale("log")
    b.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: f"{y:g}"))
    b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: f"{y:g}"))
    b.set_xlabel("slope $a$ (A/ns)"); b.set_ylabel("stored charge recovered (nC)")
    b.set_xlim(0.05, 5); b.set_ylim(30, 6000)
    b.set_title("(b) the set at the slopes of an application", loc="left")
    b.legend(loc="lower right", fontsize=7)
    save(fig, "fig_deadtime")


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    line("The body diode of a power MOSFET as a Lauritzen-Ma diode: worked examples")
    line("")
    rows = section_floor()
    c = convert_ipt()
    sims = section_sim(c)
    section_app(c)
    section_others()
    put("PEndPct", P_END*100, 0)
    fig_floor(rows); fig_coss(c); fig_test(c, sims); fig_bracket(c); fig_deadtime(c)
    with open(os.path.join(HERE, "body_diode_values.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by body_diode_calc.py: do not edit by hand\n")
        for k, v in MAC.items():
            f.write(f"\\newcommand{{\\bd{k}}}{{{v}}}\n")
    line("")
    line(f"{len(MAC)} macros written to body_diode_values.tex, figures in figures/, {time.time() - t_start:.0f} s")
    with open(os.path.join(HERE, "body_diode_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

"""Numbers and figures of power_diode_fundamentals.tex and of its Italian version power_diode_fundamentals_it.tex:
device physics of the power PIN diode.

The script computes every number that this note adds to those of the Lauritzen-Ma note, and draws its figures:
  1. silicon constants at 300 K used in the note (thermal voltage, diffusion constants, ambipolar constant);
  2. blocking design of the base: critical field, breakdown voltage, width and unipolar resistance against the
     doping (non-punch-through), and one punch-through design for comparison;
  3. forward conduction in high injection: carrier profile (catenary with unequal mobilities), stored charge,
     voltage drop on the modulated base (flat-profile estimate, exact expression, numerical check), junction
     voltages and forward voltage for three lifetimes, and the effect of the recombination in the emitters;
  4. relations of the charge-control chapter (storage time, charge at the current zero and recovery of the
     one-charge model on a ramp, transit time of the static profile) and the solution of the ambipolar diffusion
     equation on a current ramp, in closed form and checked against a finite-difference integration: delay of the
     edge concentration, charge left in the base when the junction blocks, position of the residual plasma, charge
     of the tail and softness, with equal and with unequal mobilities, what the datasheet conversion returns for
     such a base, and the slowest mode of the tail;
  5. numbers added to the worked examples of the two-charge model: base width that corresponds to the transit time
     (lumped estimates and distributed base), peak reverse voltage with the junction capacitance alone (closed form
     and simulation of the cell without a snubber), limit of the long ramp, shares of the recovery energy and
     overvoltage in a switching leg, trade-off between the base drop and the stored charge, slope of the
     open-circuit voltage decay, a unipolar silicon base for comparison.
The numbers of the two-charge model, of the conversion of datasheet data and of the worked examples are those of the
note lauritzen_ma_datasheet_conversion: this document reads lm_conversion_values.tex and the figures of that folder,
and this script only redraws the figure of the forward characteristic with the symbol N of the emission coefficient
used here. The module lm_conversion.py and the script lm_conversion_calc.py are imported from that folder, and the
script warns if the values file found there is not what lm_conversion_calc.py computes now.

Outputs, next to the script: power_diode_fundamentals_results.txt (log), power_diode_fundamentals_values.tex
(macros \\pd... of the note, read by both language versions), figures/fig_*.pdf (English labels) and
figures/fig_*_it.pdf (the same figures with Italian labels). Only numpy, scipy and matplotlib are needed.
Run:  python power_diode_fundamentals_calc.py"""
import contextlib, io, math, os, sys, time
sys.dont_write_bytecode = True             # no __pycache__ in the folder of the Lauritzen-Ma note
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.optimize import brentq, fsolve, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
LMDIR = os.path.normpath(os.path.join(HERE, "..", "lauritzen_ma_datasheet_conversion"))
sys.path.insert(0, LMDIR)
import lm_conversion as lm                 # noqa: E402  relations of the two-charge model
import lm_conversion_calc as lmc           # noqa: E402  worked examples, style and helpers of the figures

FIGDIR = os.path.join(HERE, "figures")

# ---------------------------------------------------------------- constants and assumed silicon data
Q = lm.Q_ELECTRON                          # C
KB = lm.K_BOLTZMANN                        # J/K
EPS_R = 11.7                               # relative permittivity of silicon
EPS = EPS_R*8.8541878128e-14               # F/cm, permittivity of silicon
T0 = 300.0                                 # K, temperature of the device-physics examples
VT = KB*T0/Q                               # V
NI = 1.0e10                                # cm^-3, intrinsic concentration at 300 K (round value)
MUN, MUP = 1400.0, 450.0                   # cm2/(V s), low-doping mobilities at 300 K (round values)
VSAT = 1.0e7                               # cm/s, saturation velocity
H_EM = 2.0e-14                             # cm4/s, recombination parameter of each emitter (assumed, typical order)
C_AUGER = 3.8e-31                          # cm6/s, sum of the Auger coefficients (indicative)
EG0 = 1.20                                 # eV, band gap of silicon extrapolated linearly to 0 K (sets d ln n_i / dT)
NA_EM = 1.0e19                             # cm^-3, doping of the two emitters in the examples
# example of the forward conduction: base of a 1200 V class punch-through diode
EX = dict(W=100e-4, ND=1.0e14, J=100.0, TAUS=(10e-6, 1e-6, 0.3e-6))    # cm, cm^-3, A/cm2, s
# blocking design: dopings of the table and of the 1700 V class example
ND_TABLE = (2e13, 5e13, 1e14, 2e14)
VBR_TARGET, ND_PT = 2000.0, 3.0e13         # V, cm^-3: breakdown voltage of the example and doping of its PT design

LOG, MAC = [], {}


def line(s=""):
    print(s)
    LOG.append(s)


def put(name, value, d=1):
    """macro \\pd<name> of the note (letters only); numbers with d decimals and a decimal point."""
    assert name.isalpha(), name
    MAC[name] = value if isinstance(value, str) else lmc.rnd(value, d)


def sci(x, d=1):
    """number in the form m x 10^e for the note."""
    e = int(math.floor(math.log10(abs(x))))
    return f"{x/10**e:.{d}f}\\times10^{{{e}}}"


# ---------------------------------------------------------------- relations of the note
DN, DP = VT*MUN, VT*MUP                    # cm2/s, Einstein relation
DA = 2*DN*DP/(DN + DP)                     # cm2/s, ambipolar diffusion constant in high injection
BM = (MUN - MUP)/(MUN + MUP)               # mobility asymmetry B


def da_hot(Tc):
    """Ambipolar diffusion constant in cm2/s at the temperature Tc in Celsius, with mu_n ~ T^-2.4 and mu_p ~ T^-2.2
    (exponents quoted from memory in the note)."""
    r = (Tc + 273.15)/T0
    dn, dp = DN*r**(1 - 2.4), DP*r**(1 - 2.2)
    return 2*dn*dp/(dn + dp)


def e_crit(ND):
    """Critical field in V/cm against the doping in cm^-3 (power law of the ionisation coefficients, Fulop)."""
    return 4010.0*ND**0.125


def npt_design(ND):
    """Non-punch-through base: (critical field, breakdown voltage, width at breakdown, unipolar resistance)."""
    Ec = e_crit(ND)
    V = EPS*Ec**2/(2*Q*ND)
    w = 2*V/Ec
    return Ec, V, w, w/(Q*MUN*ND)


def nd_for_vbr(V):
    """Doping of the non-punch-through base with the breakdown voltage V."""
    return brentq(lambda N: npt_design(N)[1] - V, 1e12, 1e16, xtol=1e6)


def pt_width(V, ND):
    """Width of a punch-through base of doping ND with the breakdown voltage V (trapezoidal field)."""
    Ec, k = e_crit(ND), Q*ND/(2*EPS)
    return (Ec - math.sqrt(Ec**2 - 4*k*V))/(2*k)


def ideal_coef(beta, B=BM):
    """Coefficients (A, C) of the profile p = N0*(A cosh(x/La) + C sinh(x/La)) with ideal emitters,
    N0 = J*tau/(2 q La); anode junction at x = -d, cathode junction at x = +d, beta = d/La."""
    return 1/math.sinh(beta), -B/math.cosh(beta)


def emitter_coef(beta, g, B=BM):
    """The same coefficients with recombination in the two emitters, J_n(-d) = q h p(-d)^2 and
    J_p(+d) = q h p(+d)^2; g = 2 q h N0^2/J. With J_n = J b/(1+b) + q D_a p' and J_p = J/(1+b) - q D_a p'."""
    ch, sh = math.cosh(beta), math.sinh(beta)

    def bc(v):
        A, C = v
        return (-A*sh + C*ch - g*(A*ch - C*sh)**2 + (1 + B), A*sh + C*ch + g*(A*ch + C*sh)**2 - (1 - B))
    A, C = fsolve(bc, ideal_coef(beta, B), xtol=1e-13)
    assert max(abs(v) for v in bc((A, C))) < 1e-9
    return float(A), float(C)


def profile(u, A, C):
    """Carrier profile in units of N0 against u = x/La."""
    return A*np.cosh(u) + C*np.sinh(u)


def vm_parts(beta, B=BM):
    """Voltage drop on the modulated base with ideal emitters, in units of V_T: (ohmic part, Dember part).
    The profile is p_m cosh((x - x_m)/La) with tanh(x_m/La) = B tanh(beta); the integral of 1/cosh gives two
    arctangents, which combine into the single arctangent of the form written in the note."""
    th = math.tanh(beta)
    u0 = math.atanh(B*th)
    root = math.sqrt(1 - (B*th)**2)
    ohm = (1 - B*B)*math.sinh(beta)/root*(math.atan(math.sinh(beta - u0)) + math.atan(math.sinh(beta + u0)))
    assert abs(ohm - 2*(1 - B*B)*math.sinh(beta)/root*math.atan(root*math.sinh(beta))) < 1e-9*max(1.0, ohm)
    dem = B*math.log((1 + B*th*th)/(1 - B*th*th))
    return ohm, dem


def vm_numeric(beta, A, C, B=BM):
    """The same drop by numerical integration of the field E = [J/q - (Dn - Dp) p']/((mun + mup) p), in units
    of V_T: with p in units of N0 and x in units of La, E*La/V_T = [(1 - B^2) - B p'] / p."""
    f = lambda u: ((1 - B*B) - B*(A*math.sinh(u) + C*math.cosh(u)))/(A*math.cosh(u) + C*math.sinh(u))
    return quad(f, -beta, beta, epsabs=1e-12, epsrel=1e-12)[0]


def conduction(W, ND, J, tau, h=0.0):
    """Forward conduction of the base at the current density J and the lifetime tau; h = 0: ideal emitters."""
    d = W/2
    La = math.sqrt(DA*tau)
    beta = d/La
    N0 = J*tau/(2*Q*La)                                      # scale of the profile
    A, C = emitter_coef(beta, 2*Q*h*N0**2/J) if h else ideal_coef(beta)
    pL, pR = N0*float(profile(-beta, A, C)), N0*float(profile(beta, A, C))
    um = math.atanh(-C/A)                                    # position of the minimum, in units of La
    pmin = N0*float(profile(um, A, C))
    Qs = 2*Q*N0*A*La*math.sinh(beta)                         # stored charge per unit area; tau*J with ideal emitters
    VM = VT*vm_numeric(beta, A, C)
    vj1 = VT*math.log(pL*ND/NI**2)                           # p+ n- junction
    vj2 = VT*math.log(pR/ND)                                 # n- n+ junction
    Jem = Q*h*(pL**2 + pR**2)                                # recombination current of the two emitters
    assert abs((Qs/tau + Jem)/J - 1) < 1e-8                  # charge balance J = Q/tau + emitter recombination
    return dict(La=La, beta=beta, A=A, C=C, N0=N0, pL=pL, pR=pR, pmin=pmin, xm=um*La, pbar=Qs/(Q*W), VM=VM,
                VMflat=W**2/((MUN + MUP)*tau), VMthin=2*VT*beta**2, vj1=vj1, vj2=vj2, VF=vj1 + vj2 + VM, Qs=Qs,
                tau_eff=Qs/J, Jem=Jem, TMstat=tau*(beta/math.tanh(beta) - 1))


# ---------------------------------------------------------------- 1. constants
def section_constants():
    line("1. Silicon at 300 K (assumed round values)")
    line(f"   V_T = {VT*1e3:.2f} mV, n_i = {NI:.1e} cm-3, mu_n = {MUN:g}, mu_p = {MUP:g} cm2/Vs")
    line(f"   D_n = {DN:.2f}, D_p = {DP:.2f}, D_a = {DA:.2f} cm2/s; b = mu_n/mu_p = {MUN/MUP:.2f}, B = {BM:.3f}, "
         f"1 - B^2 = {1 - BM*BM:.3f}")
    put("Vt", VT*1e3, 2); put("Mun", MUN, 0); put("Mup", MUP, 0); put("Musum", MUN + MUP, 0)
    put("Dn", DN, 0); put("Dp", DP, 0); put("Da", DA, 0); put("DaOne", DA, 1); put("Bratio", MUN/MUP, 1)
    put("Bmob", BM, 2); put("OneMinusBsq", 1 - BM*BM, 2); put("SigmaRatio", (MUN + MUP)/MUN, 2)
    line(f"   permittivity: {EPS_R:g} eps_0 = {EPS:.3e} F/cm")
    put("EpsR", EPS_R, 1); put("EpsFcm", EPS*1e12, 2)
    for tau, tag in ((1e-6, "One"), (10e-6, "Ten")):
        La = math.sqrt(DA*tau)
        line(f"   tau = {tau*1e6:g} us: L_a = {La*1e4:.1f} um")
        put("La" + tag, La*1e4, 0)
    # drop on the modulated base for d >> La: the exact expression tends to V_T*sqrt(1 - B^2)*(pi/2)*exp(d/La);
    # the handbook form drops the square root of the denominator, V_T*(1 - B^2)*(pi/2)*exp(d/La), and is written for
    # b = 3: (3*pi/8) V_T exp(d/La). The note compares the form as it is printed, with the coefficient 3*pi/8
    c_asym, c_hand = math.sqrt(1 - BM*BM)*math.pi/2, 3*math.pi/8
    line(f"   large-width asymptote of V_M: {c_asym:.3f} V_T exp(d/La); handbook form (3 pi/8) V_T exp(d/La) = "
         f"{c_hand:.3f} V_T exp(d/La), written for b = 3 ((1 - B^2) pi/2 = {(1 - BM*BM)*math.pi/2:.3f} with b = {MUN/MUP:.2f})")
    put("VmAsymCoef", c_asym, 2); put("VmHandCoef", c_hand, 2); put("ThreePiEight", 3*math.pi/8, 2)
    # intrinsic concentration against temperature: n_i ~ T^1.5 exp(-EG0/(2 k T))
    dln = lambda T: 1.5/T + EG0*Q/(2*KB*T*T)
    ratio = (398.15/298.15)**1.5*math.exp(EG0*Q/(2*KB)*(1/298.15 - 1/398.15))
    # ambipolar diffusion constant at the junction temperatures of the worked examples (mobility exponents from memory)
    for Tc, tag in ((125.0, "A"), (150.0, "B")):
        line(f"   D_a at {Tc:g} C with mu_n ~ T^-2.4, mu_p ~ T^-2.2: {da_hot(Tc):.1f} cm2/s")
        put("DaHot" + tag, da_hot(Tc), 0)
    line(f"   n_i doubles every {math.log(2)/dln(300.0):.1f} K at 300 K and every {math.log(2)/dln(400.0):.1f} K at 400 K; "
         f"n_i(125 C)/n_i(25 C) = {ratio:.0f}")
    put("NiDoubleLo", math.log(2)/dln(300.0), 0); put("NiDoubleHi", math.log(2)/dln(400.0), 0)
    put("NiRatio", round(ratio, -1), 0); put("EgZero", EG0, 1)
    # temperature coefficient of a junction voltage at constant current: (v - EG0/q - 3 V_T)/T
    for v, tag in ((0.60, "Lo"), (0.82, "Hi")):
        tc = (v - EG0 - 3*VT)/T0
        line(f"   junction voltage {v:.2f} V at constant current: dv/dT = {tc*1e3:.2f} mV/K")
        put("TcVj" + tag, v, 2); put("Tc" + tag, abs(tc)*1e3, 1)
    # Auger lifetime at two plasma densities
    for p, tag in ((1e17, "Lo"), (1e18, "Hi")):
        tA = 1/(C_AUGER*p*p)
        line(f"   Auger lifetime at p = {p:.0e} cm-3: {tA*1e6:.1f} us")
        put("TauAuger" + tag, tA*1e6, 0 if tA > 1e-5 else 1)
    line()


# ---------------------------------------------------------------- 2. blocking
def section_blocking():
    line("2. Blocking design of the base (non-punch-through unless stated)")
    for ND, tag in zip(ND_TABLE, ("A", "B", "C", "D")):
        Ec, V, w, R = npt_design(ND)
        line(f"   N_D = {ND:.0e} cm-3: E_c = {Ec/1e5:.2f}e5 V/cm, V_BR = {V:.0f} V, w = {w*1e4:.0f} um, "
             f"R_unipolar = {R:.2f} Ohm cm2 ({R*100:.0f} V at 100 A/cm2)")
        put("BlkNd" + tag, sci(ND, 0)); put("BlkEc" + tag, Ec/1e5, 2); put("BlkV" + tag, round(V, -1), 0)
        put("BlkW" + tag, w*1e4, 0); put("BlkR" + tag, R, 2); put("BlkDrop" + tag, round(R*100, 0), 0)
    # the laws V ~ N^-3/4, w ~ N^-7/8 that follow from E_c ~ N^1/8
    cV, cW = EPS*4010.0**2/(2*Q), EPS*4010.0/Q
    line(f"   closed forms with this permittivity: V_BR = {cV:.3e} N_D^-3/4 V, w = {cW:.3e} N_D^-7/8 cm")
    put("BlkCoefV", sci(cV, 1)); put("BlkCoefW", sci(cW, 1))
    # 1700 V class: breakdown voltage of 2 kV
    N = nd_for_vbr(VBR_TARGET)
    Ec, V, w, R = npt_design(N)
    wpt = pt_width(VBR_TARGET, ND_PT)
    Ecpt = e_crit(ND_PT)
    line(f"   V_BR = {VBR_TARGET:g} V: NPT needs N_D = {N:.2e} cm-3, w = {w*1e4:.0f} um, R = {R:.2f} Ohm cm2; "
         f"PT with N_D = {ND_PT:.0e} (E_c = {Ecpt/1e5:.2f}e5 V/cm) needs w = {wpt*1e4:.0f} um "
         f"({(1 - wpt/w)*100:.0f} % thinner); limit of a rectangular field: {VBR_TARGET/Ecpt*1e4:.0f} um")
    put("TgtV", VBR_TARGET/1e3, 0); put("TgtNd", sci(N, 1)); put("TgtW", w*1e4, 0)
    put("TgtR", R, 1); put("TgtDrop", round(R*100, -1), 0)
    put("PtNd", sci(ND_PT, 0)); put("PtW", wpt*1e4, 0); put("PtSavePct", round((1 - wpt/w)*100, 0), 0)
    put("PtEc", Ecpt/1e5, 2)
    # base of the conduction example read as a punch-through design
    Ecx = e_crit(EX["ND"])
    Vpt = Ecx*EX["W"] - Q*EX["ND"]*EX["W"]**2/(2*EPS)
    line(f"   base of the conduction example ({EX['W']*1e4:.0f} um, {EX['ND']:.0e} cm-3) as a PT design: "
         f"V_BR = {Vpt:.0f} V")
    put("ExVbr", round(Vpt, -1), 0)
    # dynamic avalanche: density of the holes that cross the space-charge region at the saturation velocity, and the
    # voltage at which a triangular field with the gradient q (N_D + p)/eps reaches the critical field
    for J, tag in ((100.0, "Lo"), (300.0, "Hi")):
        p = J/(Q*VSAT)
        Vdyn = npt_design(EX["ND"] + p)[1]
        line(f"   holes in transit at {J:g} A/cm2: p = J/(q v_sat) = {p:.2e} cm-3 ({p/EX['ND']*100:.0f} % of N_D = "
             f"{EX['ND']:.0e}); critical field reached at {Vdyn:.0f} V (static, same doping: {npt_design(EX['ND'])[1]:.0f} V)")
        put("DynP" + tag, sci(p, 1)); put("DynPct" + tag, round(p/EX["ND"]*100, 0), 0); put("DynJ" + tag, J, 0)
        put("DynVbr" + tag, round(Vdyn, -1), 0)
    # the junction at equilibrium and in the blocking state: built-in potentials, space-charge region, capacitance
    ND, W = EX["ND"], EX["W"]
    Vbi, Vbi2 = VT*math.log(NA_EM*ND/NI**2), VT*math.log(NA_EM/ND)
    wsc = lambda V: math.sqrt(2*EPS*(Vbi + V)/(Q*ND))
    VR = 600.0
    Vrt = Q*ND*W**2/(2*EPS) - Vbi
    Qc = Q*ND*(wsc(VR) - wsc(0.0))
    line(f"   built-in potentials: p+n- {Vbi:.3f} V, n-n+ {Vbi2:.3f} V (emitters {NA_EM:.0e} cm-3, base {ND:.0e} cm-3)")
    line(f"   space-charge region: {wsc(0.0)*1e4:.2f} um at 0 V (C_j = {EPS/wsc(0.0)*1e9:.2f} nF/cm2), {wsc(VR)*1e4:.1f} um at "
         f"{VR:g} V (C_j = {EPS/wsc(VR)*1e12:.0f} pF/cm2); charge from 0 to {VR:g} V: {Qc*1e6:.3f} uC/cm2; the "
         f"{W*1e4:.0f} um base is reached at {Vrt:.0f} V")
    put("Vbi", Vbi, 2); put("VbiTwo", Vbi2, 2); put("ScV", VR, 0); put("ScWZero", wsc(0.0)*1e4, 1)
    put("ScW", wsc(VR)*1e4, 0); put("ScCjZero", EPS/wsc(0.0)*1e9, 1); put("ScCj", round(EPS/wsc(VR)*1e12, -1), 0)
    put("ScQc", Qc*1e6, 2); put("ExVrt", round(Vrt, -1), 0)
    put("ScQcPct", Qc/(EX["J"]*EX["TAUS"][1])*100, 2)          # against the stored charge tau*J of the middle example
    # generation current of the space-charge region at 600 V with a generation lifetime equal to the middle lifetime
    Jgen = Q*NI*wsc(VR)/EX["TAUS"][1]
    line(f"   generation current at {VR:g} V, tau_g = {EX['TAUS'][1]*1e6:g} us, 300 K: {Jgen*1e6:.1f} uA/cm2")
    put("JgenUa", Jgen*1e6, 0)
    line()


# ---------------------------------------------------------------- 3. forward conduction
def section_conduction():
    line("3. Forward conduction of the base in high injection")
    W, ND, J = EX["W"], EX["ND"], EX["J"]
    R0 = W/(Q*MUN*ND)
    line(f"   base {W*1e4:.0f} um, N_D = {ND:.0e} cm-3, J = {J:g} A/cm2; without injection: R = {R0:.3f} Ohm cm2, "
         f"drop {R0*J:.0f} V")
    put("ExW", W*1e4, 0); put("ExNd", sci(ND, 0)); put("ExJ", J, 0); put("ExRuni", R0, 2); put("ExDropUni", R0*J, 0)
    put("ExNa", sci(NA_EM, 0)); put("Hem", sci(H_EM, 0)); put("ExRho", 1/(Q*MUN*ND), 0)
    # low injection and its end: Shockley current of the p+ n- junction with a minority lifetime of 1 us
    Lp = math.sqrt(DP*1e-6)
    Js, Jon, von = Q*NI**2*DP/(Lp*ND), Q*DP*ND/Lp, 2*VT*math.log(ND/NI)
    line(f"   low injection, tau_p = 1 us: L_p = {Lp*1e4:.1f} um, J_S = {Js:.2e} A/cm2; p(-d) = N_D at v_j1 = {von:.3f} V, "
         f"J = {Jon*1e3:.0f} mA/cm2")
    put("LpOne", Lp*1e4, 0); put("JsShockley", sci(Js, 1)); put("HiOnsetV", von, 2); put("HiOnsetJ", round(Jon*1e3, -1), 0)
    res = {}
    for tau, tag in zip(EX["TAUS"], ("Slow", "Mid", "Fast")):
        c = conduction(W, ND, J, tau)
        ohm, dem = vm_parts(c["beta"])
        assert abs(VT*(ohm + dem)/c["VM"] - 1) < 1e-9          # closed form against the integral of the field
        assert abs(c["Qs"]/(J*tau) - 1) < 1e-12                 # stored charge tau*J exactly
        e = conduction(W, ND, J, tau, H_EM)
        # slope of the characteristic with emitter recombination, in units of V_T per unit of ln J (numerical derivative)
        hh = 1e-3
        ep, em = conduction(W, ND, J*math.exp(hh), tau, H_EM), conduction(W, ND, J*math.exp(-hh), tau, H_EM)
        slope_vj = (ep["vj1"] + ep["vj2"] - em["vj1"] - em["vj2"])/(2*hh*VT)
        slope_vf = (ep["VF"] - em["VF"])/(2*hh*VT)
        # separation of the quasi-Fermi levels at the p+ edge of the base: k T ln(n p/n_i^2) with n = p + N_D
        qfl = VT*math.log(c["pL"]*(c["pL"] + ND)/NI**2)
        line(f"   tau = {tau*1e6:g} us: L_a = {c['La']*1e4:.1f} um, d/L_a = {c['beta']:.2f}; mean p = {c['pbar']:.2e}, "
             f"p(-d) = {c['pL']:.2e}, p(+d) = {c['pR']:.2e}, minimum {c['pmin']:.2e} at x = {c['xm']*1e4:+.1f} um")
        line(f"      stored charge {c['Qs']*1e6:.0f} uC/cm2; sigma/sigma0 = {(MUN + MUP)*c['pbar']/(MUN*ND):.0f}; "
             f"V_M: ohmic part {VT*ohm*1e3:.1f} mV (flat-profile estimate {c['VMflat']*1e3:.1f} mV), Dember part "
             f"{VT*dem*1e3:.1f} mV, total {c['VM']*1e3:.1f} mV (thin-base limit 2 V_T (d/La)^2 = {c['VMthin']*1e3:.1f} mV)")
        line(f"      junction voltages: p+n- {c['vj1']:.3f} V, n-n+ {c['vj2']:.3f} V; V_F = {c['VF']:.3f} V; "
             f"T_M/tau of the static profile (symmetric) = {c['TMstat']/tau:.3f}")
        line(f"      with emitter recombination (h = {H_EM:.0e} cm4/s each): p(-d) = {e['pL']:.2e}, p(+d) = {e['pR']:.2e}, "
             f"stored charge {e['Qs']*1e6:.1f} uC/cm2, tau_eff = {e['tau_eff']*1e6:.2f} us ({e['tau_eff']/tau*100:.0f} % of "
             f"tau), emitter share of J = {e['Jem']/J*100:.0f} %, V_M = {e['VM']*1e3:.1f} mV, V_F = {e['VF']:.3f} V")
        line(f"      slope with emitter recombination: d(v_j1 + v_j2)/d ln J = {slope_vj:.2f} V_T (2/(1 + J_E/J) = "
             f"{2/(1 + e['Jem']/J):.2f}), d V_F/d ln J = {slope_vf:.2f} V_T; mean p = {e['pbar']:.2e}")
        line(f"      quasi-Fermi levels at the p+ edge (ideal emitters): separation {qfl:.3f} eV against q v_j1 = "
             f"{c['vj1']:.3f} eV; Auger lifetime at the mean concentration {1/(C_AUGER*c['pbar']**2)*1e6:.1f} us")
        put("Ex" + tag + "Tau", tau*1e6, 1 if tau < 1e-6 else 0); put("Ex" + tag + "La", c["La"]*1e4, 0)
        put("Ex" + tag + "Beta", c["beta"], 2); put("Ex" + tag + "Pbar", sci(c["pbar"], 1))
        put("Ex" + tag + "Pl", sci(c["pL"], 1)); put("Ex" + tag + "Pr", sci(c["pR"], 1))
        put("Ex" + tag + "Pmin", sci(c["pmin"], 1)); put("Ex" + tag + "Qs", c["Qs"]*1e6, 0)
        put("Ex" + tag + "SigRatio", round((MUN + MUP)*c["pbar"]/(MUN*ND), -1), 0)
        put("Ex" + tag + "VmFlat", c["VMflat"]*1e3, 0); put("Ex" + tag + "Vm", c["VM"]*1e3, 0)
        put("Ex" + tag + "VmOhm", VT*ohm*1e3, 0); put("Ex" + tag + "VmDem", VT*dem*1e3, 0)
        put("Ex" + tag + "VmThin", c["VMthin"]*1e3, 0)
        put("Ex" + tag + "VjOne", c["vj1"], 2); put("Ex" + tag + "VjTwo", c["vj2"], 2); put("Ex" + tag + "Vf", c["VF"], 2)
        put("Ex" + tag + "TmStat", c["TMstat"]/tau, 2)
        put("Ex" + tag + "EmQs", e["Qs"]*1e6, 0); put("Ex" + tag + "EmTauEff", e["tau_eff"]*1e6, 2)
        put("Ex" + tag + "EmTauPct", e["tau_eff"]/tau*100, 0); put("Ex" + tag + "EmPct", e["Jem"]/J*100, 0)
        put("Ex" + tag + "EmVf", e["VF"], 2); put("Ex" + tag + "EmVm", e["VM"]*1e3, 0)
        put("Ex" + tag + "EmPl", sci(e["pL"], 1)); put("Ex" + tag + "EmPr", sci(e["pR"], 1))
        put("Ex" + tag + "EmPbar", sci(e["pbar"], 1)); put("Ex" + tag + "EmSlopeVj", slope_vj, 2)
        put("Ex" + tag + "EmSlopeVf", slope_vf, 2); put("Ex" + tag + "QflOne", qfl, 2)
        put("Ex" + tag + "TauAuger", 1/(C_AUGER*c["pbar"]**2)*1e6, 0)
        res[tag] = dict(ideal=c, em=e, tau=tau)
    # the drop on the modulated base against d/La: exact value, thin-base limit, ohmic part alone, asymptote,
    # handbook form of the asymptote
    c_asym, c_hand = math.sqrt(1 - BM*BM)*math.pi/2, 3*math.pi/8
    for beta, tag in ((0.25, None), (0.5, "Half"), (1.0, "One"), (2.0, "Two"), (3.0, "Three"), (4.0, "Four"), (5.0, None)):
        ohm, dem = vm_parts(beta)
        tot = ohm + dem
        thin, flat = 2*beta*beta, (1 - BM*BM)*2*beta*beta
        asym, hand = c_asym*math.exp(beta), c_hand*math.exp(beta)
        line(f"   d/L_a = {beta:g}: V_M/V_T = {tot:.3f} (ohmic {ohm:.3f}, Dember {dem:.3f}); thin-base limit "
             f"{thin:.3f} ({(thin/tot - 1)*100:+.0f} %), flat-profile estimate {flat:.3f} ({(flat/tot - 1)*100:+.0f} %), "
             f"asymptote {asym:.3f} ({(asym/tot - 1)*100:+.0f} %), handbook form {hand:.3f} ({(hand/tot - 1)*100:+.0f} %)")
        if tag:
            put("Vm" + tag + "Vt", tot, 2 if tot < 10 else 1); put("Vm" + tag + "Mv", tot*VT*1e3, 0)
            put("VmThinErr" + tag + "Pct", abs(thin/tot - 1)*100, 0)
            put("VmAsymErr" + tag + "Pct", abs(asym/tot - 1)*100, 0); put("VmHandErr" + tag + "Pct", abs(hand/tot - 1)*100, 0)
            put("VmThinSgn" + tag, f"{(thin/tot - 1)*100:+.0f}"); put("VmAsymSgn" + tag, f"{(asym/tot - 1)*100:+.0f}")
            put("VmHandSgn" + tag, f"{(hand/tot - 1)*100:+.0f}")
            put("VmFlatLow" + tag + "Pct", (1 - flat/tot)*100, 0)        # how far the flat-profile estimate is below V_M
    put("VmFlatLowPct", BM*BM*100, 0)                         # the flat-profile estimate misses the Dember part
    put("DemberLimit", BM*math.log((1 + BM)/(1 - BM)), 2)
    # forward voltage against d/La at constant current and base width: position of the minimum (ideal emitters).
    # V_F/V_T = const - 2 ln(beta) + ln(coth^2 - B^2 tanh^2) + V_M/V_T, the same for every current and width
    g = lambda be: (-2*math.log(be) + math.log(1/math.tanh(be)**2 - (BM*math.tanh(be))**2) + sum(vm_parts(be)))
    bmin = minimize_scalar(g, bounds=(0.3, 3.0), method="bounded", options=dict(xatol=1e-10)).x
    tmin = (W/2)**2/(DA*bmin**2)
    cmin = conduction(W, ND, J, tmin)
    line(f"   V_F at constant current and width is lowest at d/L_a = {bmin:.3f} (tau = {tmin*1e6:.2f} us for this base, "
         f"V_F = {cmin['VF']:.3f} V); within 10 mV of the minimum for d/L_a from "
         f"{brentq(lambda be: g(be) - g(bmin) - 0.010/VT, 0.3, bmin):.2f} to "
         f"{brentq(lambda be: g(be) - g(bmin) - 0.010/VT, bmin, 3.0):.2f}")
    put("VfMinBeta", bmin, 2); put("VfMinTau", tmin*1e6, 1); put("VfMin", cmin["VF"], 2)
    put("VfFlatLo", brentq(lambda be: g(be) - g(bmin) - 0.010/VT, 0.3, bmin), 1)
    put("VfFlatHi", brentq(lambda be: g(be) - g(bmin) - 0.010/VT, bmin, 3.0), 1)
    # quasi-neutrality: Debye length and dielectric relaxation time of the plasma of the middle example
    pb = res["Mid"]["ideal"]["pbar"]
    sig = Q*(MUN + MUP)*pb
    LD = math.sqrt(EPS*VT/(Q*2*pb))
    line(f"   plasma of the middle example: sigma = {sig:.1f} S/cm, dielectric relaxation time {EPS/sig*1e15:.0f} fs, "
         f"Debye length {LD*1e7:.0f} nm; base without injection: {1/(Q*MUN*ND):.0f} Ohm cm, relaxation {EPS*1e12/(Q*MUN*ND):.0f} ps")
    put("ExMidSigma", sig, 0); put("RelaxFs", round(EPS/sig*1e15, -1), 0); put("DebyeNm", LD*1e7, 0)
    put("RelaxPs", round(EPS*1e12/(Q*MUN*ND), -1), 0)
    # diffusion transit times of the base of the example
    d = W/2
    line(f"   transit times of the base: d^2/(3 D_a) = {d*d/(3*DA)*1e9:.0f} ns, 4 d^2/(pi^2 D_a) = {4*d*d/(math.pi**2*DA)*1e9:.0f} ns")
    put("ExTmPar", round(d*d/(3*DA)*1e9, -1), 0); put("ExTmMode", round(4*d*d/(math.pi**2*DA)*1e9, -1), 0)
    # emitter recombination against the current density, lifetime of the middle example
    tau = EX["TAUS"][1]
    for Jx, tag in ((10.0, "Lo"), (100.0, "Mid"), (300.0, "Hi")):
        e = conduction(W, ND, Jx, tau, H_EM)
        line(f"   tau = {tau*1e6:g} us, J = {Jx:g} A/cm2 with emitter recombination: tau_eff = {e['tau_eff']*1e6:.2f} us, "
             f"emitter share {e['Jem']/Jx*100:.0f} %")
        put("EmJ" + tag, Jx, 0); put("EmJ" + tag + "TauEff", e["tau_eff"]*1e6, 2); put("EmJ" + tag + "Pct", e["Jem"]/Jx*100, 0)
    line()
    return res


# ---------------------------------------------------------------- 4. charge control
def section_charge_control():
    line("4. Checks of the charge-control relations")
    tau = 1.0
    # storage time with one charge, and its two-charge form in the limit T_M -> 0
    for r in (1.0, 4.0):
        ts = tau*math.log(1 + r)
        ts2, _, _ = lm.step_recovery(r, 1.0, tau, 1e-9)
        line(f"   I_F/I_R = {r:g}: storage time {ts:.4f} tau (two-charge model with T_M -> 0: {ts2:.4f} tau)")
        assert abs(ts2/ts - 1) < 1e-6
    put("TsOne", math.log(2.0), 2); put("TsFour", math.log(5.0), 2)
    # charge at the current zero of a ramp, fraction of the static charge
    for lam, tag in ((0.2, "Fast"), (1.0, "Mid"), (5.0, "Slow")):
        frac = (1 - math.exp(-lam))/lam
        line(f"   ramp with I_F/(a tau) = {lam:g}: charge at the current zero = {frac*100:.0f} % of tau*I_F")
        put("RampFrac" + tag, frac*100, 0); put("RampLam" + tag, lam, 1)
    # recovery of the one-charge model on the same ramps: peak and recovered charge (limit T_M -> 0 of the two-charge model)
    for lam, tag in ((0.2, "Fast"), (1.0, "Mid"), (5.0, "Slow")):
        y = float(lm.ramp_peak_norm(lam, 0.0))
        assert abs(y - (1 - math.exp(-(lam + y)))) < 1e-12
        line(f"   one charge, I_F/(a tau) = {lam:g}: I_rr = {y:.3f} a tau = {y/lam:.3f} I_F, "
             f"Q_rr = {y*y/(2*lam)*100:.0f} % of tau*I_F; fast-ramp estimate sqrt(2 a tau I_F) = {math.sqrt(2*lam):.3f} a tau")
        put("RampY" + tag, y, 2); put("RampQrr" + tag, y*y/(2*lam)*100, 0); put("RampIrrIf" + tag, y/lam, 2)
    # transit time of the parabolic profile against the static profile and the slowest mode
    line(f"   T_M: parabolic profile d^2/(3 D); slowest mode 4 d^2/(pi^2 D), ratio {12/math.pi**2:.3f}")
    put("TmModeRatioPct", (12/math.pi**2 - 1)*100, 0)
    # unequal mobilities: charge left in the base when the p+ edge empties
    bq = MUN/MUP
    put("TmAsymFactor", 1 + 3*BM, 1)
    line(f"   unequal mobilities, parabolic profile: charge at the blocking instant = (1 + 3B) d^2/(3 D_a) |i|, 1 + 3B = {1 + 3*BM:.2f}")
    # distributed base on a long ramp, closed form: delay of the p+ edge, effective transit time (charge at the blocking
    # instant over the peak), softness of the tail, and what the conversion of the note returns from I_rr and Q_rr
    line("   distributed base on a long ramp (ideal emitters, constant lifetime), closed form:")
    for beta, tag in ((0.25, "Quarter"), (0.5, "Half"), (1.0, "One"), (1.5, "OneHalf"), (2.0, "Two"), (3.0, "Three")):
        r, r1 = dist_ramp(beta, bq), dist_ramp(beta, 1.0)
        tm, tm1 = 1/r["ta"] - 1, 1/r1["ta"] - 1
        assert abs(r["qblk"]/r["ta"] - tm) < 1e-12                 # charge at the blocking instant over the peak
        assert abs(r1["ta"] - 0.5*(1 + 2*beta/math.sinh(2*beta))) < 1e-12
        tau_f, tm_f, _ = lm.lm_from_irr_qrr(1.0, None, r["ta"], r["ta"]**2/2 + r["qtail"], long_ramp=True)
        line(f"      d/L_a = {beta:g}: t_a = {r['ta']:.4f} tau; T_M = tau (tau/t_a - 1) = {tm:.4f} tau = {3*tm/beta**2:.3f} "
             f"d^2/(3 D_a) (equal mobilities: {3*tm1/beta**2:.3f}); softness {r['s']:.3f} (equal mobilities {r1['s']:.3f}); "
             f"residual plasma peaks at x/d = {r['xpk']:+.2f}")
        line(f"         conversion of I_rr and Q_rr: tau = {tau_f:.3f} of the true one, T_M = {tm_f:.3f} tau = "
             f"{3*tm_f/beta**2:.2f} d^2/(3 D_a): base width by d^2/(3 D_a), {math.sqrt(3*tm_f/beta**2):.2f} times the true one")
        put("DistTa" + tag, r["ta"], 2); put("DistTm" + tag, tm, 3); put("DistTmGeom" + tag, 3*tm/beta**2, 2)
        put("DistS" + tag, r["s"], 2); put("DistTauFit" + tag, tau_f, 2); put("DistTmFit" + tag, tm_f, 3)
        put("DistWidth" + tag, math.sqrt(3*tm_f/beta**2), 2); put("TmPeakPos" + tag, r["xpk"], 2)
        if beta <= 1.0:
            put("TmBlockSym" + tag, 3*tm1/beta**2, 2); put("TmBlockAsym" + tag, 3*tm/beta**2, 1)
            # check of the closed form against the finite-difference solution, on a ramp of 5 lifetimes
            for b in (1.0, bq):
                pde, xp, spde = blocking_transit(beta, b, 5.0)
                rf = dist_ramp(beta, b, 5.0)
                assert abs(pde/(rf["qblk"]/rf["ta"]) - 1) < 3e-3, (beta, b, pde, rf["qblk"]/rf["ta"])
                assert abs(xp - rf["xpk"]) < 0.02, (beta, b, xp, rf["xpk"])
                assert abs(spde/rf["s"] - 1) < 2e-3, (beta, b, spde, rf["s"])
                line(f"         finite differences, ramp of 5 lifetimes, b = {b:.2f}: charge at the blocking instant over the "
                     f"peak {pde:.4f} tau (closed form {rf['qblk']/rf['ta']:.4f}), softness {spde:.4f} ({rf['s']:.4f})")
        if tag == "Quarter":
            assert abs(3*tm1/beta**2 - 1) < 0.02 and abs(3*tm/beta**2/(1 + 3*BM) - 1) < 0.05
            assert abs(r["xpk"] - BM) < 0.02                           # thin base: maximum of the parabola at x = B d
    # thick base: delay tau/2, T_M = tau, softness 1; the conversion returns 3/4 of the lifetime and T_M = 3 tau/8
    r = dist_ramp(9.0, bq)
    tau_f, tm_f, _ = lm.lm_from_irr_qrr(1.0, None, r["ta"], r["ta"]**2/2 + r["qtail"], long_ramp=True)
    line(f"      d/L_a = 9: t_a = {r['ta']:.5f} tau, softness {r['s']:.5f}, conversion: tau = {tau_f:.5f}, T_M = {tm_f:.5f} tau "
         f"(limits 1/2, 1, 3/4, 3/8)")
    assert abs(r["ta"] - 0.5) < 1e-5 and abs(r["s"] - 1) < 1e-3 and abs(tau_f - 0.75) < 1e-3 and abs(tm_f - 0.375) < 1e-3
    for b, tag in ((bq, ""), (1.0, "Sym")):
        res = minimize_scalar(lambda be: -dist_ramp(be, b)["s"], bounds=(1.0, 4.0), method="bounded", options=dict(xatol=1e-7))
        one = brentq(lambda be: dist_ramp(be, b)["s"] - 1, 0.5, res.x)
        line(f"      b = {b:.2f}: largest softness on a long ramp {-res.fun:.4f} at d/L_a = {res.x:.2f}; softness 1 at d/L_a = {one:.2f}")
        put("DistSmax" + tag, -res.fun, 2); put("DistSmaxBeta" + tag, res.x, 1); put("DistSoneBeta" + tag, one, 1)
    # base width at which the maximum of the residual plasma is at the centre of the base
    cen = brentq(lambda be: dist_ramp(be, bq)["xpk"], 1.0, 1.5, xtol=1e-4)
    line(f"      b = {bq:.2f}: the residual plasma has its maximum at the centre of the base for d/L_a = {cen:.2f}")
    put("TmPeakZeroBeta", cen, 1)
    # slowest mode of the blocked base: p = 0 at the p+ edge, gradients at the two edges in the ratio b: cos(2 k d) = -1/b
    kd = 0.5*math.acos(-1/bq)
    line(f"   tail mode with b = {bq:.2f}: k d = {kd:.3f}, T_M = {1/kd**2:.2f} d^2/D_a = {3/kd**2:.2f} times d^2/(3 D_a) "
         f"(equal mobilities: {4/math.pi**2:.3f} d^2/D_a)")
    put("TmTailCoef", 1/kd**2, 1); put("TmTailFactor", 3/kd**2, 1); put("TmTailCoefSym", 4/math.pi**2, 2)
    put("TmTailAngle", 2*kd, 2)
    line()


def dist_ramp(beta, b, lam=None, nmax=400):
    """Distributed base on a current ramp: ambipolar diffusion equation with ideal emitters and a constant lifetime, in
    closed form. Units: tau = 1, L_a = 1, x in [-beta, beta] with the p+ edge at -beta; ramp of unit slope (currents in
    a tau, charges in a tau^2) that starts from the steady state at j = lam = I_F/(a tau); lam = None is the long ramp.
    For a current e^(s t) the profile is g(x; w) = cosh(w x)/(2 w sinh(w beta)) - B sinh(w x)/(2 w cosh(w beta)) with
    w = sqrt(1 + s). The response to the ramp is therefore (lam - t) g0 - g1, with g0 = g(w = 1) and g1 = (1/2) dg/dw at
    w = 1, plus the transient of the start of the ramp, a series of the modes cos(k_n (x + beta)), k_n = n pi/(2 beta).
    Returns: ta, time from the current zero to the instant the concentration at the p+ edge reaches zero, which is also
    I_rr/(a tau); tb, the same instant from the start of the ramp; qblk, charge in the base at that instant; qtail,
    charge extracted afterwards, with the p+ edge at zero and the gradients at the two edges in the ratio b (from the
    time integral P of the profile: P'' - P = -p(tb), P(-beta) = 0, P'(beta) = -P'(-beta)/b); the softness
    s = qtail/(ta^2/2); xpk, position of the maximum of the residual plasma in units of d."""
    B = (b - 1)/(b + 1)
    sh, ch = math.sinh(beta), math.cosh(beta)
    g0 = lambda x: np.cosh(x)/(2*sh) - B*np.sinh(x)/(2*ch)
    g1 = lambda x: 0.25*(x*np.sinh(x)/sh - np.cosh(x)*(sh + beta*ch)/sh**2
                         - B*x*np.cosh(x)/ch + B*np.sinh(x)*(ch + beta*sh)/ch**2)
    k = np.arange(nmax)*math.pi/(2*beta)
    rate = 1 + k*k
    coef = np.where(np.arange(nmax) % 2 == 0, 1.0, B)/beta
    coef[0] = 1/(2*beta)

    def p(x, t):                            # t from the start of the ramp (from the current zero on the long ramp)
        if lam is None:
            return float(-t*g0(x) - g1(x))
        return float((lam - t)*g0(x) - g1(x) - np.sum(np.cos(k*(x + beta))*coef*np.exp(-rate*t)/rate**2))

    if lam is None:
        ta = tb = float(-g1(-beta)/g0(-beta))
    else:
        hi = lam + 1e-6
        while p(-beta, hi) > 0:
            hi = lam + 2*(hi - lam)
        tb = brentq(lambda t: p(-beta, t), lam, hi, xtol=1e-14, rtol=1e-13)
        ta = tb - lam
    qblk = 1 - tb if lam is None else lam - tb + 1 - math.exp(-tb)       # equation of the charge control, exact
    integ = quad(lambda x: math.cosh(beta - x)*p(x, tb), -beta, beta, epsabs=1e-13, epsrel=1e-12, limit=400)[0]
    qtail = 2/(1 + B)*integ/(math.cosh(2*beta) + 1/b)
    xs = np.linspace(-beta, beta, 801)
    j = int(np.argmax([p(x, tb) for x in xs]))
    xpk = minimize_scalar(lambda x: -p(x, tb), bounds=(xs[max(j - 1, 0)], xs[min(j + 1, 800)]), method="bounded",
                          options=dict(xatol=1e-10)).x
    return dict(ta=ta, tb=tb, qblk=qblk, qtail=qtail, s=qtail/(ta*ta/2), xpk=xpk/beta)


def blocking_transit(beta, b, lam, N=300, dt=1e-3):
    """Ambipolar diffusion equation of the base on the ramp i = I_F (1 - t/(lam tau)) from the steady state, ideal emitters.
    Units: x/d, t/tau, p/(J_F tau/(q d)). Returns Q/(|i| tau) at the instant the concentration at the p+ edge reaches zero
    (the transit time of the blocking condition q_M = T_M |i|, in lifetimes), the position of the residual maximum, and
    the softness: the charge of the tail, from the discretised operator of the blocked base, over that of the triangle."""
    h = 2.0/N
    x = -1 + h*(np.arange(N) + 0.5)
    k, B = 1.0/beta**2, (b - 1)/(b + 1)
    p = (beta/2)*(np.cosh(beta*x)/math.sinh(beta) - B*np.sinh(beta*x)/math.cosh(beta))      # catenary, integral 1
    r = k*dt/h**2
    ab = np.zeros((3, N))
    ab[0, 1:] = -0.5*r; ab[2, :-1] = -0.5*r
    ab[1] = 1 + r + 0.5*dt; ab[1, 0] = ab[1, -1] = 1 + 0.5*r + 0.5*dt

    def rhs_op(p):
        out = np.empty_like(p)
        out[1:-1] = k*(p[2:] - 2*p[1:-1] + p[:-2])/h**2
        out[0], out[-1] = k*(p[1] - p[0])/h**2, k*(p[-2] - p[-1])/h**2
        return out - p

    def src(j):                             # the two boundary fluxes: j b/(1+b) at the p+ edge, j/(1+b) at the n+ edge
        s = np.zeros(N)
        s[0], s[-1] = j*b/(1 + b)/h, j/(1 + b)/h
        return s

    def edge(p, j):                         # concentration at x = -d from the first cell and the gradient set by the current
        return p[0] + 0.5*h*j*beta**2*b/(1 + b)

    t, e0, p0 = 0.0, edge(p, 1.0), p.copy()
    while True:
        j0, j1 = 1 - t/lam, 1 - (t + dt)/lam
        p = solve_banded((1, 1), ab, p + 0.5*dt*rhs_op(p) + 0.5*dt*(src(j0) + src(j1)))
        t += dt
        e = edge(p, j1)
        if e <= 0:
            f = e0/(e0 - e)
            pb = p0 + f*(p - p0)
            jb = 1 - (t - dt + f*dt)/lam
            # blocked base, dp/dt = A2 p: zero concentration at the p+ edge (gradient from the parabola through the edge
            # and the first two cells) and flux at the n+ edge equal to 1/b of the flux at the p+ edge. The time integral
            # of the profile is -A2^-1 p_b, and the charge of the tail is the time integral of the current at the p+ edge.
            A2 = (np.diag(np.full(N, -2*k/h**2 - 1)) + np.diag(np.full(N - 1, k/h**2), 1)
                  + np.diag(np.full(N - 1, k/h**2), -1))
            A2[0, 0], A2[0, 1], A2[-1, -1] = -4*k/h**2 - 1, 4*k/(3*h**2), -k/h**2 - 1
            A2[-1, 0] -= 3*k/(b*h**2); A2[-1, 1] += k/(3*b*h**2)
            P = -np.linalg.solve(A2, pb)
            qtail = (1 + b)/b*k*(9*P[0] - P[1])/(3*h)
            return pb.sum()*h/abs(jb), x[int(np.argmax(pb))], 2*qtail/(lam*jb*jb)
        e0, p0 = e, p.copy()
        if t > lam + 20:
            raise RuntimeError("the junction does not block")


# ---------------------------------------------------------------- 5. worked examples of the two-charge model
def section_examples(exA, exB):
    line("5. Worked examples of the Lauritzen-Ma note: base width behind T_M, peak voltage with the junction capacitance, "
         "energies in a switching leg")
    bq = MUN/MUP
    for key, ex in (("A", exA), ("B", exB)):
        D, TM = ex["d"]["D"], ex["c"]["TM"]
        w_par = 2*math.sqrt(3*D*TM)                          # equal mobilities, parabolic profile (value of the note)
        w_blk = w_par/math.sqrt(1 + 3*BM)                    # blocking condition with unequal mobilities
        w_tail = math.acos(-1/bq)*math.sqrt(D*TM)            # tail mode with unequal mobilities
        line(f"   diode {key}: T_M = {TM*1e6:.3g} us, D_a = {D*1e4:g} cm2/s: base width {w_par*1e6:.0f} um (parabola, equal "
             f"mobilities), {w_blk*1e6:.0f} um (blocking condition, b = {bq:.2f}), {w_tail*1e6:.0f} um (tail mode, b = {bq:.2f})")
        put(key + "Wblock", w_blk*1e6, 0); put(key + "Wtail", w_tail*1e6, 0)
        # the diffusion constants of the examples are those of the Lauritzen-Ma note; with the power law of this note
        dlaw = da_hot(ex["d"]["T"])
        line(f"      D_a at {ex['d']['T']:g} C by the power law of this note: {dlaw:.1f} cm2/s, widths {(math.sqrt(dlaw*1e-4/D) - 1)*100:+.1f} %")
        put(key + "WlawPct", abs(math.sqrt(dlaw*1e-4/D) - 1)*100, 0)
    # the same check with the distributed base. Diode A: the base (d/L_a and lifetime) that gives the two data on the
    # test ramp; for each d/L_a the lifetime is set by the peak, and the charge follows
    d, c = exA["d"], exA["c"]
    IF, a, Irr, Qrr, D = c["IF"], c["a"], c["Irr"], c["Qrr"], d["D"]

    def base_fit(beta):
        tau = brentq(lambda tau: dist_ramp(beta, bq, IF/(a*tau))["ta"]*a*tau - Irr, 0.5*c["tau"], 3*c["tau"], xtol=1e-13)
        r = dist_ramp(beta, bq, IF/(a*tau))
        return (r["ta"]**2/2 + r["qtail"])*a*tau**2, tau, r

    bmax = minimize_scalar(lambda be: -base_fit(be)[0], bounds=(1.3, 3.0), method="bounded", options=dict(xatol=1e-5)).x
    qmax, taumax, _ = base_fit(bmax)
    line(f"   diode A with a distributed base (b = {bq:.2f}, D_a = {D*1e4:g} cm2/s): the lifetime is set by I_rr = {Irr:g} A, "
         f"and the charge follows; conversion of the note: tau = {c['tau']*1e6:.2f} us, T_M = {c['TM']*1e6:.2f} us")
    for beta in (0.5, 1.0, 1.3, bmax, 3.0, 4.0, 8.0):
        qq, tau, r = base_fit(beta)
        line(f"      d/L_a = {beta:.2f}: tau = {tau*1e6:.2f} us, Q_rr = {qq*1e6:.0f} uC ({(qq/Qrr - 1)*100:+.1f} % on the datum), "
             f"softness {r['s']:.3f}, base width {2*beta*math.sqrt(D*tau)*1e6:.0f} um")
    roots = [brentq(lambda be: base_fit(be)[0] - Qrr, lo, hi, xtol=1e-9) for lo, hi in ((1.0, bmax), (bmax, 6.0))]
    for be, tag in zip(roots, ("", "Hi")):
        qq, tau, r = base_fit(be)
        w = 2*be*math.sqrt(D*tau)
        line(f"      both data reproduced with d/L_a = {be:.3f}, tau = {tau*1e6:.2f} us, base width {w*1e6:.0f} um "
             f"(softness {r['s']:.3f}, ramp of {(IF + Irr)/(a*tau):.2f} lifetimes)")
        put("ADistBeta" + tag, be, 2); put("ADistTau" + tag, tau*1e6, 0); put("ADistW" + tag, round(w*1e6, -1), 0)
        put("ADistWmm" + tag, w*1e3, 1)
    blo = 1.3                                                    # narrowest base still close to the data
    qlo, taulo, _ = base_fit(blo)
    qinf, tauinf, _ = base_fit(8.0)
    put("ADistLoBeta", blo, 1); put("ADistLoW", round(2*blo*math.sqrt(D*taulo)*1e6, -1), 0)
    put("ADistLoWmm", 2*blo*math.sqrt(D*taulo)*1e3, 1)
    put("ADistLoPct", abs(qlo/Qrr - 1)*100, 1); put("ADistMaxPct", (qmax/Qrr - 1)*100, 1)
    put("ADistInfPct", abs(qinf/Qrr - 1)*100, 1); put("ADistTol", max(abs(qlo/Qrr - 1), abs(qinf/Qrr - 1), qmax/Qrr - 1)*100, 1)
    put("ADistTauLo", taulo*1e6, 0); put("ADistTauInf", tauinf*1e6, 0)
    assert qlo < Qrr < qmax and qinf < Qrr
    # diode B: softness of its data against the largest softness of a distributed base at the same ratio I_rr/I_F
    d, c = exB["d"], exB["c"]
    ratio, s_data = c["Irr"]/c["IF"], 2*c["tau_rr"]/c["ta"]

    def soft_at_ratio(beta, b):
        lam = brentq(lambda lam: dist_ramp(beta, b, lam)["ta"]/lam - ratio, 1e-3, 50.0, xtol=1e-11)
        return dist_ramp(beta, b, lam)["s"]

    for b, tag in ((bq, ""), (1.0, "Sym")):
        res = minimize_scalar(lambda be: -soft_at_ratio(be, b), bounds=(0.6, 4.0), method="bounded", options=dict(xatol=1e-5))
        line(f"   diode B: softness of the data {s_data:.2f}; distributed base with I_rr/I_F = {ratio:g}, b = {b:.2f}: "
             f"largest softness {-res.fun:.3f} at d/L_a = {res.x:.2f}")
        put("BDistSmax" + tag, -res.fun, 2); put("BDistSmaxBeta" + tag, res.x, 1)
        assert -res.fun < s_data
    # peak reverse voltage in the cell without a snubber: the junction capacitance alone. Closed form for a square-root
    # capacitance law and a tail much slower than the oscillation, 3 V_RM - (V_bi/2)(sqrt(9 + 12 V_RM/V_bi) - 3), and
    # simulation of the cell of the Lauritzen-Ma note without its snubber
    for key, ex in (("A", exA), ("B", exB)):
        d, c, dev = ex["d"], ex["c"], ex["chk"]["dev"]
        VR, L, s = d["VR"], d["VR"]/c["a"], 2*c["tau_rr"]/c["ta"]
        vrm, vbi = VR*(1 + 2/s), dev.get("Vj", 0.7)
        assert dev.get("m", 0.5) == 0.5
        vhat = 3*vrm - 0.5*vbi*(math.sqrt(9 + 12*vrm/vbi) - 3)
        r = lm.ramp_recovery(c["a"], c["IF"], c["tau"], c["TM"])
        sim = lm.simulate_cell(dev, c["IF"], VR, L, t_end=r["T1"] + 0.6*r["tau_rr"], points=120001)
        vr = -sim["v"]
        k = int(np.argmax(vr))
        # first maximum of the reverse voltage after the junction blocks, and the minimum that follows it
        k0 = int(np.argmax(vr > 0.5*VR))
        k1 = next(i for i in range(k0, len(vr) - 1) if vr[i] >= vr[i - 1] and vr[i] > vr[i + 1])
        k2 = next(i for i in range(k1, len(vr) - 1) if vr[i] <= vr[i - 1] and vr[i] < vr[i + 1])
        k3 = next(i for i in range(k2, len(vr) - 1) if vr[i] >= vr[i - 1] and vr[i] > vr[i + 1])
        cpk = lm.cj(-vr[k1], dev["Cj0"])
        line(f"   diode {key} in the cell without a snubber: V_RM = V_R (1 + 2/s) = {vrm:.0f} V without capacitance; with the "
             f"junction capacitance alone the first peak is {vhat:.0f} V by the closed form and {vr[k1]:.0f} V in the simulation "
             f"(then the voltage returns to {abs(vr[k2]):.0f} V and oscillates with a period of {(sim['t'][k3] - sim['t'][k1])*1e9:.0f} ns; "
             f"sqrt(L C_j) at the peak = {math.sqrt(L*float(cpk))*1e9:.1f} ns against tau_rr = {r['tau_rr']*1e9:.0f} ns); "
             f"with the snubber of the note {-ex['chk']['s']['v'].min():.0f} V")
        put(key + "VpkCjPeriod", (sim["t"][k3] - sim["t"][k1])*(1e6 if key == "A" else 1e9), 1 if key == "A" else 0)
        put(key + "VpkCj", round(vhat, -1), 0); put(key + "VpkCjSim", round(vr[k1], -1), 0)
        put(key + "VpkCjRatio", vhat/vrm, 1)
        assert abs(vr[k1]/vhat - 1) < 0.03 and k == k1
    # long ramp of diode A: slope at which the ramp of its set lasts five and three lifetimes, and error of the
    # long-ramp relations at 10 A/us
    c = exA["c"]
    for xi, tag in ((5.0, "Five"), (3.0, "Three")):
        ax = brentq(lambda a: lm.ramp_recovery(a, c["IF"], c["tau"], c["TM"])["x"] - xi, 0.1e6, 100e6, xtol=1.0)
        line(f"   diode A: the ramp of the set lasts {xi:g} lifetimes at {ax/1e6:.1f} A/us")
        put("ALongSlope" + tag, ax/1e6, 0)
    a10 = 10e6
    r = lm.ramp_recovery(a10, c["IF"], c["tau"], c["TM"])
    e_i = r["Irr"]/(a10*(c["tau"] - r["tau_rr"])) - 1
    e_q = r["Qrr"]/(0.5*a10*(c["tau"]**2 - r["tau_rr"]**2)) - 1
    line(f"   diode A at 10 A/us: ramp of {r['x']:.2f} lifetimes; exact I_rr and Q_rr are {e_i*100:+.1f} % and {e_q*100:+.1f} % "
         f"on the long-ramp relations")
    put("ATenXi", r["x"], 1); put("ATenIrrPct", abs(e_i)*100, 0); put("ATenQrrPct", abs(e_q)*100, 0)
    # diode B at its datasheet point in a switching leg with a real switch (L_sigma a << V_R): shares of the energy
    d, c = exB["d"], exB["c"]
    VR, IF, Irr, Qrr, ta, trr = d["VR"], c["IF"], c["Irr"], c["Qrr"], c["ta"], c["tau_rr"]
    s = 2*trr/ta
    e_ideal, e_diode = VR*Qrr, VR*Irr*trr
    e_on, e_load, e_qa = VR*(IF + Irr/2)*ta, VR*IF*ta, VR*Irr*ta/2
    assert abs(e_diode + e_qa - e_ideal) < 1e-9*e_ideal
    line(f"   diode B, {VR:g} V, {IF:g} A, I_rr = {Irr:g} A, t_a = {ta*1e9:.0f} ns, tau_rr = {trr*1e9:.0f} ns, s = {s:.2f}:")
    line(f"      V_R Q_rr = {e_ideal*1e3:.2f} mJ; diode with a real switch V_R I_rr tau_rr = {e_diode*1e3:.2f} mJ "
         f"(k = s/(1+s) = {s/(1 + s):.2f}); extra turn-on energy of the switch V_R (I_F + I_rr/2) t_a = {e_on*1e3:.2f} mJ "
         f"({e_load*1e3:.2f} mJ load current + {e_qa*1e3:.2f} mJ recovery current)")
    put("SwErecIdeal", e_ideal*1e3, 1); put("SwErecDiode", e_diode*1e3, 1); put("SwK", s/(1 + s), 2)
    put("SwEonExtra", e_on*1e3, 1); put("SwEonLoad", e_load*1e3, 1); put("SwEonQa", e_qa*1e3, 1)
    put("SwRatio", e_on/e_diode, 1)
    # overvoltage 2 L_sigma a/s in a switching leg, for the softness of diode B and for a snappy diode; ideal-switch cell
    Ls, a, s_snap = 50e-9, c["a"], 0.5
    line(f"      overvoltage in a leg with L_sigma = {Ls*1e9:g} nH at {a/1e6:g} A/us: {2*Ls*a/s:.0f} V with s = {s:.2f}, "
         f"{2*Ls*a/s_snap:.0f} V with s = {s_snap:g}; cell with an ideal switch (L = V_R/a = {VR/a*1e9:.0f} nH): {2*VR/s:.0f} V")
    put("SwLs", Ls*1e9, 0); put("SwOvSoft", round(2*Ls*a/s, 0), 0); put("SwOvSnap", round(2*Ls*a/s_snap, 0), 0)
    put("SwSnapS", s_snap, 1); put("SwOvIdeal", round(2*VR/s, -1), 0)
    # trade-off of the thin base: V_M tau = w^2 (mu_n + mu_p)/(4 mu_n mu_p), independent of the lifetime
    vmtau = EX["W"]**2*(MUN + MUP)/(4*MUN*MUP)
    line(f"   base of {EX['W']*1e4:.0f} um: V_M tau = w^2 (mu_n+mu_p)/(4 mu_n mu_p) = {vmtau*1e9:.1f} mV us (thin-base limit)")
    put("ExVmTau", vmtau*1e9, 0)
    # open-circuit voltage decay in high injection, lifetime of 1 us
    line(f"   open-circuit voltage decay with tau = 1 us: 2 V_T/tau = {2*VT/1e-6*1e-3:.1f} mV/us")
    put("OcvdSlope", 2*VT/1e-6*1e-3, 0)
    # a silicon unipolar base for 1200 V (non-punch-through), for the comparison with silicon carbide
    N = nd_for_vbr(1200.0)
    _, _, w, R = npt_design(N)
    line(f"   silicon unipolar base for 1200 V: N_D = {N:.2e} cm-3, w = {w*1e4:.0f} um, R = {R:.2f} Ohm cm2 ({R*100:.0f} V at 100 A/cm2)")
    put("SiUniR", R, 2); put("SiUniDrop", round(R*100, 0), 0); put("SiUniW", round(w*1e4, 0), 0)
    line()


# ---------------------------------------------------------------- figures
# Each figure is drawn twice, as in lm_conversion_calc.py: English labels (figures/fig_*.pdf) and Italian labels
# (figures/fig_*_it.pdf). lmc.L(en, it) returns the label in the current language, lmc.LANG.
def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    if lmc.LANG == "it":
        lmc.comma_ticks(fig)
    fig.savefig(os.path.join(FIGDIR, name + ("_it" if lmc.LANG == "it" else "") + ".pdf"))
    plt.close(fig)


def fig_profile(res):
    """Carrier profile of the base: catenary for the three lifetimes, and the concentrations on a logarithmic scale."""
    W, ND = EX["W"], EX["ND"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(wspace=0.36))
    b = ax[0]
    cols = (lmc.RAMP[1], lmc.RAMP[3], lmc.RAMP[5])
    for (tag, r), col, (xl, off) in zip(res.items(), cols, ((-0.40, (24, 16)), (-0.36, (24, 12)), (-0.42, (16, 6)))):
        c = r["ideal"]
        u = np.linspace(-c["beta"], c["beta"], 600)
        y = c["N0"]*profile(u, c["A"], c["C"])/c["pbar"]     # p(x)/mean
        b.plot(u*c["La"]/W, y, color=col)
        b.plot([c["xm"]/W], [c["pmin"]/c["pbar"]], "o", ms=2.8, color=col)
        k = int(np.argmin(np.abs(u*c["La"]/W - xl)))
        lmc.note(b, lmc.loc_num(f"{r['tau']*1e6:g}") + " µs", (u[k]*c["La"]/W, y[k]), off, va="bottom")
    b.axhline(1.0, color=lmc.MUTED, lw=0.6, ls=(0, (4, 3)))
    b.set_xlim(-0.5, 0.5); b.set_ylim(0, 3.5)
    b.text(-0.48, 0.12, "p$^{+}$", fontsize=7.5, ha="left", va="bottom", color=lmc.INK2)
    b.text(0.48, 0.12, "n$^{+}$", fontsize=7.5, ha="right", va="bottom", color=lmc.INK2)
    b.set_xlabel(lmc.L("position in the base, $x/w_\\mathrm{B}$", "posizione nella base, $x/w_\\mathrm{B}$")); b.set_ylabel("$p(x)/\\bar{p}$")
    b.set_title(lmc.L("(a) catenary profile for three lifetimes", "(a) profilo a catenaria per tre lifetime"), loc="left")
    # --- logarithmic scale, lifetime of the middle example
    b = ax[1]
    c = res["Mid"]["ideal"]
    u = np.linspace(-c["beta"], c["beta"], 600)
    x, y = u*c["La"]/W, c["N0"]*profile(u, c["A"], c["C"])
    xe = 0.22
    b.plot([-0.5 - xe, -0.5, -0.5], [NA_EM, NA_EM, c["pL"]], color=lmc.C_NOM)
    b.plot(x, y, color=lmc.C_NOM)
    b.plot([0.5, 0.5, 0.5 + xe], [c["pR"], NA_EM, NA_EM], color=lmc.C_NOM)
    b.plot([-0.5, 0.5], [ND, ND], color=lmc.INK, lw=0.9, ls=(0, (4, 2)))
    b.axvline(-0.5, color=lmc.MUTED, lw=0.5, ls=":"); b.axvline(0.5, color=lmc.MUTED, lw=0.5, ls=":")
    b.set_yscale("log"); b.set_ylim(1e13, 6e19); b.set_xlim(-0.5 - xe, 0.5 + xe)
    b.yaxis.set_major_locator(lmc.mticker.LogLocator(base=10, numticks=8))
    b.yaxis.set_minor_formatter(lmc.mticker.NullFormatter())
    k = int(np.argmin(np.abs(x - 0.25)))
    b.annotate("", (x[k], ND), (x[k], y[k]), arrowprops=dict(arrowstyle="<->", color=lmc.INK2, lw=0.6, shrinkA=0, shrinkB=0))
    b.text(x[k] - 0.03, math.sqrt(ND*y[k]), lmc.L("almost three\ndecades", "quasi tre\ndecadi"), fontsize=7, ha="right", va="center", color=lmc.INK2)
    k = int(np.argmin(np.abs(x + 0.2)))
    lmc.note(b, "plasma, $n \\simeq p$", (x[k], y[k]), (4, 16), va="bottom")
    lmc.note(b, "$N_\\mathrm{D}$: " + lmc.L("without injection", "senza iniezione"), (-0.25, ND), (6, -12), va="top")
    b.text(-0.5 - xe/2, 2.5e18, "p$^{+}$", fontsize=7.5, ha="center", va="center", color=lmc.INK2)
    b.text(0.5 + xe/2, 2.5e18, "n$^{+}$", fontsize=7.5, ha="center", va="center", color=lmc.INK2)
    b.set_xlabel(lmc.L("position, $x/w_\\mathrm{B}$", "posizione, $x/w_\\mathrm{B}$"))
    b.set_ylabel(lmc.L("concentration (cm$^{-3}$)", "concentrazione (cm$^{-3}$)"))
    b.set_xticks([-0.5, 0, 0.5])
    b.set_title(lmc.L("(b) concentrations", "(b) concentrazioni") + ", $\\tau$ = " + lmc.loc_num(f"{res['Mid']['tau']*1e6:g}") + " µs", loc="left")
    save(fig, "fig_profile")


def fig_vm(res):
    """Drop on the modulated base against d/La, and the forward voltage of the example against the lifetime."""
    W, ND, J = EX["W"], EX["ND"], EX["J"]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw=dict(wspace=0.30))
    b = ax[0]
    beta = np.logspace(math.log10(0.1), math.log10(4.0), 300)
    parts = np.array([vm_parts(v) for v in beta])
    exact = parts.sum(axis=1)
    b.plot(beta, 2*beta**2, color=lmc.MUTED, lw=0.8, ls=(0, (4, 3)))
    sel = beta > 1.3
    b.plot(beta[sel], math.sqrt(1 - BM*BM)*math.pi/2*np.exp(beta[sel]), color=lmc.MUTED, lw=0.8, ls=(0, (1, 2)))
    b.plot(beta, parts[:, 0], color=lmc.C_WEAK, lw=1.0)
    b.plot(beta, exact, color=lmc.C_NOM)
    for tag, r in res.items():
        c = r["ideal"]
        b.plot([c["beta"]], [c["VM"]/VT], "o", ms=3.5, color=lmc.INK, zorder=5)
        b.annotate(lmc.loc_num(f"{r['tau']*1e6:g}") + " µs", (c["beta"], c["VM"]/VT), xytext=(-5, 4),
                   textcoords="offset points", fontsize=7, ha="right", va="bottom")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(0.1, 4.0); b.set_ylim(0.008, 60)
    lmc.plain_log(b)
    lmc.note(b, "$2\\,(d/L_\\mathrm{a})^2$", (2.6, 2*2.6**2), (10, -14), va="top")
    lmc.note(b, "$\\propto e^{d/L_\\mathrm{a}}$", (1.45, math.sqrt(1 - BM*BM)*math.pi/2*math.exp(1.45)), (-18, 10),
             ha="right", va="bottom")
    k = int(np.argmin(np.abs(beta - 0.2)))
    lmc.note(b, lmc.L("ohmic part\nonly", "sola parte\nohmica"), (beta[k], parts[k, 0]), (12, -16), va="top")
    b.set_xlabel(lmc.L("half width of the base, $d/L_\\mathrm{a}$", "semilarghezza della base, $d/L_\\mathrm{a}$"))
    b.set_ylabel("$V_\\mathrm{M}/V_\\mathrm{T}$")
    b.set_title(lmc.L("(a) drop on the modulated base", "(a) caduta sulla base modulata"), loc="left")
    # --- forward voltage against the lifetime
    b = ax[1]
    taus = np.logspace(math.log10(0.15e-6), math.log10(20e-6), 160)
    cs = [conduction(W, ND, J, t) for t in taus]
    es = [conduction(W, ND, J, t, H_EM) for t in taus]
    vj = np.array([c["vj1"] + c["vj2"] for c in cs]); vm = np.array([c["VM"] for c in cs])
    vfe = np.array([e["VF"] for e in es])
    b.plot(taus*1e6, vj, color=lmc.RAMP[1]); b.plot(taus*1e6, vm, color=lmc.C_WEAK)
    b.plot(taus*1e6, vfe, color=lmc.INK, lw=0.9, ls=(0, (4, 2)))
    b.plot(taus*1e6, vj + vm, color=lmc.INK)
    for tag, r in res.items():
        b.plot([r["tau"]*1e6], [r["ideal"]["VF"]], "o", ms=3.5, color=lmc.INK, zorder=5)
    b.set_xscale("log"); b.set_xlim(taus[0]*1e6, taus[-1]*1e6); b.set_ylim(0, 1.45)
    lmc.plain_log(b, "x")
    k = int(np.argmin(np.abs(taus - 0.45e-6)))
    lmc.note(b, "$v_\\mathrm{j1} + v_\\mathrm{j2}$", (taus[k]*1e6, vj[k]), (0, -6), ha="center", va="top", arrow=False)
    k = int(np.argmin(np.abs(taus - 2.2e-6)))
    lmc.note(b, "$V_\\mathrm{F}$", (taus[k]*1e6, vj[k] + vm[k]), (-2, 22), ha="center", va="bottom")
    k = int(np.argmin(np.abs(taus - 9e-6)))
    lmc.note(b, lmc.L("with recombination\nin the emitters", "con ricombinazione\nnegli emitter"), (taus[k]*1e6, vfe[k]), (-6, -34),
             ha="right", va="top")
    k = int(np.argmin(np.abs(taus - 0.3e-6)))
    lmc.note(b, "$V_\\mathrm{M}$", (taus[k]*1e6, vm[k]), (14, 8), va="bottom")
    b.set_xlabel("lifetime $\\tau$ (µs)"); b.set_ylabel(lmc.L("voltage (V)", "tensione (V)"))
    b.set_title("(b) forward voltage " + lmc.L("at", "a") + " " + f"{J:g}" + " A/cm$^2$", loc="left")
    save(fig, "fig_vm")


def fig_forward(exA, exB):
    """Forward characteristic of the two worked examples (figure of lm_conversion_calc.py with the symbol N)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.27))
    d, c = exA["d"], exA["c"]
    i = np.linspace(1.0, 2000.0, 800)
    b = ax[0]
    b.plot(i, d["VT0"] + d["RT"]*i, color=lmc.INK, lw=0.9, ls=(0, (4, 2)))
    for nn, col in ((1.0, lmc.C_WEAK), (2.0, lmc.C_NOM)):
        b.plot(i, lm.forward_voltage(i, exA["fits"][nn][0], nn, c["VT"], exA["fits"][nn][1]), color=col)
    b.plot([d["I1"], d["I2"]], [d["VT0"] + d["RT"]*d["I1"], d["VT0"] + d["RT"]*d["I2"]], "o", ms=3.5, color=lmc.INK)
    k = int(np.argmin(np.abs(i - 130)))
    lmc.note(b, "$N$ = 1", (i[k], float(lm.forward_voltage(i[k], exA["fits"][1.0][0], 1.0, c["VT"], exA["fits"][1.0][1]))),
             (16, -12))
    k = int(np.argmin(np.abs(i - 60)))
    lmc.note(b, "$N$ = 2", (i[k], float(lm.forward_voltage(i[k], exA["fits"][2.0][0], 2.0, c["VT"], exA["fits"][2.0][1]))),
             (24, -15))
    lmc.note(b, "$V_\\mathrm{T0} + r_\\mathrm{T}\\,i$", (230, d["VT0"] + d["RT"]*230), (6, 40), va="bottom")
    b.set_xlim(0, 2000); b.set_ylim(0.5, 2.1)
    b.set_xlabel("forward current (A)"); b.set_ylabel("forward voltage (V)")
    b.set_title(lmc.L("(a) diode A: straight line of the datasheet", "(a) diodo A: retta del datasheet"), loc="left")
    d, c = exB["d"], exB["c"]
    i = np.linspace(0.2, 200.0, 800)
    b = ax[1]
    b.plot(i, exB["VT0"] + exB["rT"]*i, color=lmc.INK, lw=0.9, ls=(0, (4, 2)))
    b.plot(i, lm.forward_voltage(i, exB["IS"], exB["n"], c["VT"], exB["RS"]), color=lmc.C_NOM)
    b.plot([p[0] for p in d["VF"]], [p[1] for p in d["VF"]], "o", ms=3.5, color=lmc.INK)
    lmc.note(b, lmc.L("model", "modello") + ", $N$ = " + lmc.loc_num("%.2f" % exB["n"]),
             (150, float(lm.forward_voltage(150.0, exB["IS"], exB["n"], c["VT"], exB["RS"]))), (-10, 14), ha="right")
    lmc.note(b, lmc.L("line through the\ntwo upper points", "retta per i due\npunti superiori"), (12, exB["VT0"] + exB["rT"]*12), (20, -24))
    b.set_xlim(0, 200); b.set_ylim(0.5, 2.9)
    b.set_xlabel("forward current (A)"); b.set_ylabel("forward voltage (V)")
    b.set_title(lmc.L("(b) diode B: three points of the curve", "(b) diodo B: tre punti della curva"), loc="left")
    save(fig, "fig_forward")


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    line("Power PIN diode, device physics: numbers of power_diode_fundamentals.tex")
    line()
    section_constants()
    section_blocking()
    res = section_conduction()
    section_charge_control()
    # worked examples of the two-charge model: computed by the script of the Lauritzen-Ma note, without its output
    with contextlib.redirect_stdout(io.StringIO()):
        lmc.section_general()
        exA, exB = lmc.section_a(), lmc.section_b()
    section_examples(exA, exB)
    with open(os.path.join(HERE, "power_diode_fundamentals_results.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(LOG) + "\n")
    with open(os.path.join(HERE, "power_diode_fundamentals_values.tex"), "w", encoding="utf-8") as fo:
        fo.write("% generated by power_diode_fundamentals_calc.py: do not edit by hand\n")
        for k, v in MAC.items():
            fo.write(f"\\newcommand{{\\pd{k}}}{{{v}}}\n")
    for lang in ("en", "it"):              # figures/fig_*.pdf, then figures/fig_*_it.pdf
        lmc.LANG = lang
        fig_profile(res)
        fig_vm(res)
        fig_forward(exA, exB)
    lmc.LANG = "en"
    # the numbers of Parts III and IV are read from the values file of the Lauritzen-Ma note: check it is current
    stale = []
    with open(os.path.join(LMDIR, "lm_conversion_values.tex"), encoding="utf-8") as fi:
        txt = fi.read()
    for k, v in lmc.MAC.items():
        if f"\\newcommand{{\\lm{k}}}{{{v}}}" not in txt and k not in ("FigRho", "FigLam"):
            stale.append(k)
    if stale:
        print("WARNING: lm_conversion_values.tex differs from what lm_conversion_calc.py computes now: " + ", ".join(stale))
    print(f"done in {time.time() - t0:.0f} s: {len(MAC)} macros, figures in {FIGDIR}")


if __name__ == "__main__":
    main()

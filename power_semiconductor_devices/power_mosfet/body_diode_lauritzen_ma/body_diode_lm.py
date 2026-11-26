"""Relations that the body diode of a power MOSFET adds to the Lauritzen-Ma conversion.
Companion module of body_diode_lauritzen_ma.tex; used by body_diode_calc.py. The relations of the two-charge
model itself (ramp solution, conversion from Irr and Qrr, forward fits) are imported from lm_conversion.py in
research/power_semiconductor_devices/diode/lauritzen_ma_datasheet_conversion.

What is added here:
  - the output capacitance C_oss(v) as a power law, its charge Q_oss(V) and two ways to fit it to a datasheet;
  - the turn-off ramp that starts from a base charged for a finite time only (dead time shorter than the lifetime);
  - the charge injected by the body diode while the channel conducts in the third quadrant;
  - a simulated commutation cell with the complete device: diffusion current, C_oss(v), channel, series resistance.
Notation as in lm_conversion.py: a = -di/dt > 0, IF forward current, Irr peak reverse current (> 0), Qrr recovered
charge, tau lifetime, TM transit time, tau_rr = tau*TM/(tau + TM). SI units. Needs numpy and scipy only.
Run:  python body_diode_lm.py      (self-test: the dead-time ramp against a step-by-step integration)"""
import math
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
LMDIR = os.path.normpath(os.path.join(HERE, "..", "..", "diode", "lauritzen_ma_datasheet_conversion"))
sys.path.insert(0, LMDIR)
import lm_conversion as lm  # noqa: E402


# ====================================================================== output capacitance
def coss(v, C0, Vj, m):
    """C_oss(v) = C0/(1 + v/Vj)^m for a reverse voltage v > 0; constant C0 for v <= 0 (conducting diode)."""
    v = np.asarray(v, float)
    return C0/np.power(1 + np.maximum(v, 0.0)/Vj, m)


def qoss(V, C0, Vj, m):
    """Charge of C_oss(v) between zero and the reverse voltage V (closed form of the power law)."""
    V = np.asarray(V, float)
    if abs(m - 1) < 1e-12:
        return C0*Vj*np.log1p(np.maximum(V, 0.0)/Vj)
    return C0*Vj/(1 - m)*(np.power(1 + np.maximum(V, 0.0)/Vj, 1 - m) - 1)


def eoss(V, C0, Vj, m):
    """Energy stored in C_oss(v) at the reverse voltage V, integral of v*C(v) dv."""
    V = float(V)
    if abs(m - 1) < 1e-12 or abs(m - 2) < 1e-12:
        from scipy.integrate import quad
        return quad(lambda x: x*float(coss(x, C0, Vj, m)), 0.0, V)[0]
    u = 1 + V/Vj
    return C0*Vj**2*((u**(2 - m) - 1)/(2 - m) - (u**(1 - m) - 1)/(1 - m))


def fit_coss_point_charge(C_at_V, V, Q_at_V, Vj, m_range=(0.05, 1.5)):
    """(C0, m) of the power law from the capacitance C_at_V and the charge Q_at_V, both at the voltage V, with Vj
    given. The two data fix the two parameters; the ratio Q/(C*V) depends on m only."""
    target = Q_at_V/(C_at_V*V)

    def gap(m):
        C0 = C_at_V*(1 + V/Vj)**m
        return float(qoss(V, C0, Vj, m))/(C_at_V*V) - target
    m = brentq(gap, *m_range, xtol=1e-12)
    return C_at_V*(1 + V/Vj)**m, m


def fit_coss_points(points, Vj, m_fixed=None):
    """(C0, m) of the power law through points [(V, C), ...] of the C_oss curve, least squares in log C against
    log(1 + V/Vj). With m_fixed only C0 is fitted (geometric mean)."""
    V = np.array([p[0] for p in points], float)
    C = np.array([p[1] for p in points], float)
    x, y = np.log1p(V/Vj), np.log(C)
    if m_fixed is not None:
        return float(np.exp(np.mean(y + m_fixed*x))), m_fixed
    A = np.vstack([np.ones_like(x), -x]).T
    (lnC0, m), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.exp(lnC0)), float(m)


# ====================================================================== dead time shorter than the lifetime
def charge_after_dead_time(IF, tau, td, q_start=0.0):
    """Middle charge after the diode has carried IF for the time td, starting from q_start (charge-control
    equation with constant current): q_M = tau*IF + (q_start - tau*IF)*exp(-td/tau)."""
    return tau*IF + (q_start - tau*IF)*math.exp(-td/tau)


def ramp_recovery_from_charge(a, IF, tau, TM, qM0):
    """Turn-off on the ramp i = IF - a*t when the base holds the middle charge qM0 at the start of the ramp
    (qM0 = tau*IF is the steady state and gives lm.ramp_recovery). Exact for the model.
    Returns a dict: Irr, ta, T1, tau_rr, Qrr, x = T1/tau, F = Irr over the long-ramp peak, s, q0 (charge at the
    current zero), qM0."""
    trr = lm.tail_time(tau, TM)
    c = qM0 - tau*IF - a*tau**2                            # amplitude of the transient of q_M

    def qE(t):
        return (tau + TM)*(IF - a*t) + a*tau**2 + c*math.exp(-t/tau)
    t0 = IF/a
    if qE(0.0) <= 0:                                       # nothing to recover: the junction blocks at once
        return dict(Irr=0.0, ta=0.0, T1=t0, tau_rr=trr, Qrr=0.0, x=t0/tau, F=0.0, s=float("nan"),
                    q0=0.0, qM0=qM0)
    hi = t0
    while qE(hi) > 0:
        hi += tau
    T1 = brentq(qE, t0, hi, xtol=1e-16*tau, rtol=1e-14)
    Irr = a*T1 - IF
    ta = Irr/a
    return dict(Irr=Irr, ta=ta, T1=T1, tau_rr=trr, Qrr=0.5*Irr*ta + Irr*trr, x=T1/tau,
                F=Irr/(a*(tau - trr)), s=2*trr/ta if ta > 0 else float("inf"),
                q0=tau*0.0 + a*tau**2 + c*math.exp(-t0/tau), qM0=qM0)


def ramp_recovery_dead_time(a, IF, tau, TM, td, q_start=0.0):
    """Recovery after a dead time td: the channel turns off at t = -td, the diode takes the current IF at once and
    the ramp starts at t = 0. q_start is the middle charge while the channel conducted (small, see
    channel_injection)."""
    return ramp_recovery_from_charge(a, IF, tau, TM, charge_after_dead_time(IF, tau, td, q_start))


def integrate_dead_time(a, IF, tau, TM, td, q_start=0.0, dt=None):
    """Independent check: the charge equations integrated step by step (RK4) through the dead time and the ramp.
    Returns (Irr, Qrr)."""
    dt = dt or min(tau, TM, td if td > 0 else tau)/4000
    qM, t = q_start, -td
    while t < -1e-15*tau:                                   # dead time: constant current IF
        h = min(dt, -t)
        k1 = IF - qM/tau
        k2 = IF - (qM + h/2*k1)/tau
        k3 = IF - (qM + h/2*k2)/tau
        k4 = IF - (qM + h*k3)/tau
        qM, t = qM + h/6*(k1 + 2*k2 + 2*k3 + k4), t + h
    t = 0.0

    def f(tt, q):
        return IF - a*tt - q/tau
    while True:                                             # ramp, conducting junction
        k1 = f(t, qM)
        k2 = f(t + dt/2, qM + dt/2*k1)
        k3 = f(t + dt/2, qM + dt/2*k2)
        k4 = f(t + dt, qM + dt*k3)
        qn = qM + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if qn + TM*(IF - a*(t + dt)) <= 0:
            e0, e1 = qM + TM*(IF - a*t), qn + TM*(IF - a*(t + dt))
            h = dt*e0/(e0 - e1)
            qM, t = qM + (qn - qM)*h/dt, t + h
            break
        qM, t = qn, t + dt
    Irr = a*t - IF
    Q = 0.5*a*(t - IF/a)**2
    lam = 1/tau + 1/TM
    while qM/TM > 1e-9*max(Irr, 1e-30):                     # tail
        k1 = -lam*qM
        k2 = -lam*(qM + dt/2*k1)
        k3 = -lam*(qM + dt/2*k2)
        k4 = -lam*(qM + dt*k3)
        qn = qM + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        Q += 0.5*(qM + qn)/TM*dt
        qM = qn
    return Irr, Q


# ====================================================================== channel in parallel
def channel_injection(I, Rds, IS, n, VT, RS=0.0):
    """Reverse conduction with the channel on. The current I (source to drain) splits between the channel,
    v = Rds*i_ch, and the diode, v = n*VT*ln(1 + i_d/IS) + RS*i_d. Returns (i_d, v): diode current and
    source-drain voltage. The stored charge is tau*i_d."""
    def gap(i_d):
        return n*VT*math.log1p(i_d/IS) + RS*i_d - Rds*(I - i_d)
    if gap(0.0) >= 0:
        return 0.0, Rds*I
    i_d = brentq(gap, 0.0, I, xtol=1e-18*I, rtol=1e-14)
    return i_d, Rds*(I - i_d)


# ====================================================================== commutation cell with the complete device
def simulate_body_cell(dev, IF, VR, L, td=None, q_start=None, t_end=None, points=20001, rtol=1e-9):
    """Commutation cell of the datasheet test: the device carries IF (source to drain, channel off) and at t = 0 a
    switch applies VR through L, so the current falls with about VR/L. The device is the diffusion current of the
    two-charge model, C_oss(v) and the channel in parallel, in series with RS.
    dev: dict with tau, TM, IS, n, VT, RS, C0, Vj, m and optionally Rr (leakage), Rds (channel on-resistance).
    td = None: the diode starts from its steady state at IF (the datasheet test). td given: the channel conducts
    until t = -td with the middle charge q_start (default: the steady-state charge of channel_injection), then the
    current moves to the diode for the dead time td and the ramp starts at t = 0.
    States: current of the switch, middle charge, voltage across the device (positive in reverse).
    Returns a dict with t, i (device current, forward positive), v (device voltage, forward positive), i_dif,
    i_c (current of C_oss), i_ch, qM, qE."""
    tau, TM, IS, n, VT, RS = dev["tau"], dev["TM"], dev["IS"], dev["n"], dev["VT"], dev["RS"]
    C0, Vj, m = dev["C0"], dev["Vj"], dev["m"]
    Rr, Rds = dev.get("Rr", 1e9), dev.get("Rds", None)
    nVT, qS = n*VT, (tau + TM)*IS

    def rhs(t, y, g_ch):
        isw, qM, vr = y                                     # vr = reverse voltage across the junction (= -v_j)
        iD = IF - isw                                       # device current, forward positive
        vj = -vr
        qE = qS*math.expm1(min(vj/nVT, 80.0))
        idif = (qE - qM)/TM
        C = float(coss(vr, C0, Vj, m))
        ic = iD - idif - vj*g_ch - vj/Rr                    # current left for C_oss
        return [(VR - vj + RS*iD)/L, idif - qM/tau, -ic/C]

    # --- initial state
    vj0 = nVT*math.log1p(IF/IS)
    if td is None:
        y0, segs = [0.0, tau*IF, -vj0], []
    else:
        g_on = 1/Rds
        if q_start is None:
            i_d, v = channel_injection(IF, Rds, IS, n, VT, RS)
            q_start = tau*i_d
        y0 = [0.0, q_start, -(nVT*math.log1p(max(i_d, 1e-30)/IS))]
        segs = [(-td, 0.0, 0.0)]                            # dead time: channel off, no ramp yet (switch open)
    r = lm.ramp_recovery(VR/L, IF, tau, TM)
    if t_end is None:
        t_end = r["T1"] + 14*r["tau_rr"]
    t_all, y_all = [], []
    if segs:                                                # dead time: hold the switch open (isw stays 0)
        ta_, tb_, _ = segs[0]
        tt = np.linspace(ta_, tb_, max(int(points*td/t_end), 400))

        def rhs_dead(t, y):
            isw, qM, vr = y
            vj = -vr
            qE = qS*math.expm1(min(vj/nVT, 80.0))
            idif = (qE - qM)/TM
            C = float(coss(vr, C0, Vj, m))
            ic = IF - idif - vj/Rr
            return [0.0, idif - qM/tau, -ic/C]
        sol = solve_ivp(rhs_dead, [ta_, tb_], y0, method="Radau", t_eval=tt, rtol=rtol,
                        atol=[1e-9*IF, 1e-9*tau*IF, 1e-8])
        if not sol.success:
            raise RuntimeError(sol.message)
        t_all.append(sol.t[:-1]); y_all.append(sol.y[:, :-1])
        y0 = list(sol.y[:, -1])
    tt = np.linspace(0.0, t_end, points)
    sol = solve_ivp(lambda t, y: rhs(t, y, 0.0), [0.0, t_end], y0, method="Radau", t_eval=tt, rtol=rtol,
                    atol=[1e-9*IF, 1e-9*tau*IF, 1e-8])
    if not sol.success:
        raise RuntimeError(sol.message)
    t_all.append(sol.t); y_all.append(sol.y)
    t = np.concatenate(t_all)
    isw, qM, vr = np.concatenate(y_all, axis=1)
    iD = IF - isw
    vj = -vr
    qE = qS*np.expm1(np.minimum(vj/nVT, 80.0))
    idif = (qE - qM)/TM
    ic = iD - idif - vj/Rr
    return dict(t=t, i=iD, v=vj + RS*iD, vr=vr, i_dif=idif, i_c=ic, qM=qM, qE=qE, steps=sol.nfev)



def fit_coss_two_voltages(C_at_V1, V1, Q_at_V2, V2, Vj, m_range=(0.02, 1.8)):
    """(C0, m) of the power law from the capacitance at V1 and the charge from zero to V2, with Vj given.
    Generalises fit_coss_point_charge to two different voltages (e.g. C_oss at 50 V and Q_oss, or C_oss(tr)*V, up
    to the test voltage)."""
    def gap(m):
        C0 = C_at_V1*(1 + V1/Vj)**m
        return float(qoss(V2, C0, Vj, m)) - Q_at_V2
    m = brentq(gap, *m_range, xtol=1e-12)
    return C_at_V1*(1 + V1/Vj)**m, m


# ====================================================================== the recovery test seen at the terminals
# After the junction blocks (peak of the diffusion current Irr at T1) the voltage across the device rises to VR and
# C_oss takes the charge Q_oss(VR). What the terminals show during this phase depends on the driver of the test:
#   "ramp"    the driver keeps the slope a: the terminal current goes on along the ramp until v = VR;
#   "plateau" the driver holds its current (Miller plateau): the terminal current stays at Irr until v = VR.
# In both cases the charge of the window that ends when the voltage reaches VR is Q_dif + Q_oss(VR), and at that
# instant the terminal current drops to the remaining diffusion tail, which then decays with tau_rr.
def terminal_ramp(a, Irr, tau_rr, Qoss_VR):
    """Driver that keeps the slope. Returns a dict: Irrm (terminal peak), Tc (duration of the voltage rise, from
    the diffusion peak), i_res (diffusion current left when the voltage reaches VR), Qrr (terminal charge, equal
    to Q_dif + Q_oss(VR)), Qtri (charge of the terminal triangle Irrm^2/(2a)), trr for the conventions:
    'line' (straight line through 90 % and 25 % of the fall: the fall is a step, so t_b = 0), 'p25' and 'p10'
    (instant at which the current is below 25 % and 10 % of Irrm)."""
    ta = Irr/a
    def gap(Tc):   # capacitive charge taken between T1 and T1 + Tc
        return 0.5*a*((ta + Tc)**2 - ta**2) - Irr*tau_rr*(1 - math.exp(-Tc/tau_rr)) - Qoss_VR
    hi = math.sqrt(2*Qoss_VR/a) + ta + 1e-30
    while gap(hi) < 0:
        hi *= 2
    Tc = brentq(gap, 0.0, hi, xtol=1e-20, rtol=1e-14)
    Irrm = a*(ta + Tc)
    i_res = Irr*math.exp(-Tc/tau_rr)
    out = dict(Irrm=Irrm, Tc=Tc, i_res=i_res, Qrr=0.5*Irrm**2/a + i_res*tau_rr, Qtri=0.5*Irrm**2/a,
               Qdif=0.5*Irr**2/a + Irr*tau_rr, trr_line=ta + Tc)
    for name, p in (("trr_p25", 0.25), ("trr_p10", 0.10)):
        out[name] = ta + Tc + (tau_rr*math.log(i_res/(p*Irrm)) if i_res > p*Irrm else 0.0)
    return out


def terminal_plateau(a, Irr, tau_rr, Qoss_VR):
    """Driver that holds its current: flat top at Irr until the voltage reaches VR. Same keys as terminal_ramp;
    here Irrm = Irr and the straight-line t_b is the length of the flat top plus the step."""
    ta = Irr/a
    def gap(Tc):
        return Irr*Tc - Irr*tau_rr*(1 - math.exp(-Tc/tau_rr)) - Qoss_VR
    hi = Qoss_VR/Irr + tau_rr + 1e-30
    while gap(hi) < 0:
        hi *= 2
    Tc = brentq(gap, 0.0, hi, xtol=1e-20, rtol=1e-14)
    i_res = Irr*math.exp(-Tc/tau_rr)
    out = dict(Irrm=Irr, Tc=Tc, i_res=i_res, Qrr=0.5*Irr*ta + Irr*Tc + i_res*tau_rr, Qtri=0.5*Irr*ta,
               Qdif=0.5*Irr**2/a + Irr*tau_rr, trr_line=ta + Tc)
    for name, p in (("trr_p25", 0.25), ("trr_p10", 0.10)):
        out[name] = ta + Tc + (tau_rr*math.log(i_res/(p*Irr)) if i_res > p*Irr else 0.0)
    return out


def capacitive_only(a, Qoss_VR):
    """Recovery of the output capacitance alone (no stored charge), driver that keeps the slope: terminal peak,
    duration of the ramp after the current zero and charge."""
    Irrm = math.sqrt(2*a*Qoss_VR)
    return dict(Irrm=Irrm, trr=Irrm/a, Qrr=Qoss_VR)


def invert_terminal(a, Qoss_VR, Qrr=None, Irrm=None, trr=None, driver="plateau", convention="p25"):
    """Peak of the diffusion current Irr and tail time constant tau_rr from two terminal data of the datasheet
    (two of Qrr, Irrm, trr) under one driver model. Qrr is always reduced by Q_oss(VR): Q_dif = Qrr - Q_oss(VR).
    Returns (Irr, tau_rr, info) or raises ValueError when the data have no solution under that model."""
    f = terminal_plateau if driver == "plateau" else terminal_ramp
    key = {"p25": "trr_p25", "p10": "trr_p10", "line": "trr_line"}[convention]
    if Qrr is not None:
        Qdif = Qrr - Qoss_VR
        if Qdif <= 0:
            raise ValueError("Qrr does not exceed Q_oss(VR): no stored charge can be resolved")
        Irr_max = math.sqrt(2*a*Qdif)*(1 - 1e-12)      # tau_rr > 0 needs Irr^2/(2a) < Q_dif

        def tail_of(Irr):
            return (Qdif - 0.5*Irr**2/a)/Irr
        if Irrm is not None:
            if driver == "plateau":
                if Irrm >= Irr_max:
                    raise ValueError("Irrm^2/(2a) exceeds Qrr - Q_oss(VR): no tail is left under the plateau model")
                Irr = Irrm
                return Irr, tail_of(Irr), f(a, Irr, tail_of(Irr), Qoss_VR)
            def gap(Irr):
                return f(a, Irr, tail_of(Irr), Qoss_VR)["Irrm"] - Irrm
            lo, hi = 1e-9*Irr_max, Irr_max
            if gap(lo) > 0 or gap(hi) < 0:
                raise ValueError("Irrm and Qrr - Q_oss(VR) have no solution under the ramp model")
            Irr = brentq(gap, lo, hi, xtol=1e-15, rtol=1e-13)
            return Irr, tail_of(Irr), f(a, Irr, tail_of(Irr), Qoss_VR)
        if trr is not None:
            def gap(Irr):
                return f(a, Irr, tail_of(Irr), Qoss_VR)[key] - trr
            lo, hi = 1e-6*Irr_max, Irr_max
            if gap(lo)*gap(hi) > 0:
                raise ValueError("trr and Qrr - Q_oss(VR) have no solution under this model")
            Irr = brentq(gap, lo, hi, xtol=1e-15, rtol=1e-13)
            return Irr, tail_of(Irr), f(a, Irr, tail_of(Irr), Qoss_VR)
    if Irrm is not None and trr is not None:
        if driver == "plateau":
            Irr = Irrm
            def gap(tau_rr):
                return f(a, Irr, tau_rr, Qoss_VR)[key] - trr
            lo, hi = 1e-13, trr
            if gap(lo)*gap(hi) > 0:
                raise ValueError("Irrm and trr have no solution under the plateau model")
            tau_rr = brentq(gap, lo, hi, xtol=1e-20, rtol=1e-13)
            return Irr, tau_rr, f(a, Irr, tau_rr, Qoss_VR)
        raise ValueError("(Irrm, trr) under the ramp model: trr = Irrm/a is fixed, the pair carries one datum only")
    raise ValueError("two of Qrr, Irrm, trr are needed")


# ====================================================================== simulated recovery test with a driver
def simulate_test(dev, IF, VR, a, driver="plateau", Ls=10e-9, Rs=0.5, t_end=None, points=12001, rtol=1e-8):
    """The reverse recovery test of a MOSFET body diode: the device (channel off) carries IF; the driver takes
    the current over with the slope a; after the junction blocks the driver either keeps the slope ("ramp") or
    holds its current ("plateau") until the device voltage reaches VR; then the device sits across the voltage
    source VR through the stray inductance Ls and the resistance Rs, and the current rings back.
    dev: tau, TM, IS, n, VT, RS, C0, Vj, m, optionally Rr. States: middle charge, reverse voltage across the
    junction (and C_oss), and in the last phase the terminal current.
    Returns a dict with t, i (terminal current, forward positive), v (terminal voltage, reverse positive), i_dif,
    i_c, qM, T1 (junction blocks), T2 (voltage reaches VR)."""
    tau, TM, IS, n, VT, RS = dev["tau"], dev["TM"], dev["IS"], dev["n"], dev["VT"], dev["RS"]
    C0, Vj, m, Rr = dev["C0"], dev["Vj"], dev["m"], dev.get("Rr", 1e9)
    nVT, qS = n*VT, (tau + TM)*IS

    def diode(qM, vr):
        vj = -vr
        qE = qS*math.expm1(min(vj/nVT, 80.0))
        return (qE - qM)/TM, qE

    def rhs_imposed(t, y, i_of_t):                          # phases with the current imposed by the driver
        qM, vr = y
        i = i_of_t(t)
        idif, qE = diode(qM, vr)
        return [idif - qM/tau, (idif - i + vr/Rr)/float(coss(vr, C0, Vj, m))]

    def rhs_clamp(t, y):                                    # device across VR through Ls, Rs
        qM, vr, i = y
        idif, qE = diode(qM, vr)
        return [idif - qM/tau, (idif - i + vr/Rr)/float(coss(vr, C0, Vj, m)),
                -(VR - vr + (RS + Rs)*i)/Ls]

    t0 = IF/a
    r = lm.ramp_recovery(a, IF, tau, TM)
    T_guess = r["T1"] + 20*r["tau_rr"] + 2*math.sqrt(2*float(qoss(VR, C0, Vj, m))/a)
    t_end = t_end or T_guess + 12*math.sqrt(Ls*float(coss(VR, C0, Vj, m))) + 10*r["tau_rr"]
    atol = [1e-9*tau*IF, 1e-7]
    vj0 = nVT*math.log1p(IF/IS)
    y0 = [tau*IF, -vj0]
    out_t, out_i, out_qM, out_vr = [], [], [], []

    # --- phase 1: ramp, until the junction voltage reverses (vr crosses zero upward)
    ramp = lambda t: IF - a*t
    ev_block = lambda t, y: y[1]
    ev_block.terminal, ev_block.direction = True, 1
    sol = solve_ivp(lambda t, y: rhs_imposed(t, y, ramp), [0.0, t_end], y0, method="Radau", rtol=rtol, atol=atol,
                    events=ev_block, dense_output=True, max_step=min(tau, TM)/2)
    T1 = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    if T1 is None:
        raise RuntimeError("the junction did not block within t_end")
    tt = np.linspace(0.0, T1, max(int(points*T1/t_end), 200))
    yy = sol.sol(tt)
    out_t.append(tt); out_qM.append(yy[0]); out_vr.append(yy[1]); out_i.append(ramp(tt))
    y1 = list(sol.sol(T1))
    # --- phase 2: voltage rise, current imposed by the driver (ramp kept or held), until vr = VR
    i_hold = ramp(T1)
    i_of_t = ramp if driver == "ramp" else (lambda t: i_hold)
    ev_vr = lambda t, y: y[1] - VR
    ev_vr.terminal, ev_vr.direction = True, 1
    sol = solve_ivp(lambda t, y: rhs_imposed(t, y, i_of_t), [T1, t_end], y1, method="Radau", rtol=rtol, atol=atol,
                    events=ev_vr, dense_output=True, max_step=min(tau, TM)/2)
    T2 = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    if T2 is None:
        raise RuntimeError("the voltage did not reach VR within t_end")
    tt = np.linspace(T1, T2, max(int(points*(T2 - T1)/t_end), 200))
    yy = sol.sol(tt)
    out_t.append(tt[1:]); out_qM.append(yy[0][1:]); out_vr.append(yy[1][1:])
    out_i.append(np.asarray([i_of_t(x) for x in tt[1:]], float))
    y2 = list(sol.sol(T2)) + [i_of_t(T2)]
    # --- phase 3: clamp
    tt = np.linspace(T2, t_end, max(int(points*(t_end - T2)/t_end), 400))
    sol = solve_ivp(rhs_clamp, [T2, t_end], y2, method="Radau", rtol=rtol, atol=atol + [1e-9*IF], t_eval=tt,
                    max_step=min(tau, TM, math.sqrt(Ls*float(coss(VR, C0, Vj, m))))/2)
    if not sol.success:
        raise RuntimeError(sol.message)
    out_t.append(sol.t[1:]); out_qM.append(sol.y[0][1:]); out_vr.append(sol.y[1][1:]); out_i.append(sol.y[2][1:])
    t = np.concatenate(out_t); qM = np.concatenate(out_qM); vr = np.concatenate(out_vr); i = np.concatenate(out_i)
    qE = qS*np.expm1(np.minimum(-vr/nVT, 80.0))
    idif = (qE - qM)/TM
    return dict(t=t, i=i, v=vr - RS*i, vr=vr, i_dif=idif, i_c=i - idif + vr/Rr, qM=qM, qE=qE, T1=T1, T2=T2, t0=t0)


def measure_test(sim, p_end=0.02):
    """Datasheet quantities of a simulated test: lm.measure (peak, charge down to p_end of the peak, trr by the
    straight line and at 10 %) plus the charge to the first current zero after the peak and trr at 25 %."""
    t, i = sim["t"], sim["i"]
    m = lm.measure(t, i, p_end=p_end)
    kp = int(np.argmin(i))
    k0 = int(np.argmax(i < 0))
    t0 = t[k0 - 1] + (t[k0] - t[k0 - 1])*i[k0 - 1]/(i[k0 - 1] - i[k0])
    after = np.where(i[kp:] >= 0)[0]
    if len(after):
        k = kp + int(after[0])
        tz = t[k - 1] + (t[k] - t[k - 1])*(-i[k - 1])/(i[k] - i[k - 1])
        sel = (t >= t0) & (t <= tz)
        m["Qrr_zero"] = lm.trapezoid(np.concatenate(([0.0], -i[sel], [0.0])), np.concatenate(([t0], t[sel], [tz])))
        m["t_zero"] = tz - t0
    else:
        m["Qrr_zero"], m["t_zero"] = m["Qrr"], float("nan")
    tail = -i[kp:]
    k = int(np.argmax(tail <= 0.25*m["Irr"]))
    m["trr_25"] = (t[kp + k - 1] + (t[kp + k] - t[kp + k - 1])*(tail[k - 1] - 0.25*m["Irr"])/(tail[k - 1] - tail[k]) - t0) if k > 0 else float("nan")
    return m


# ====================================================================== self-test
if __name__ == "__main__":
    print("Body diode relations: self-test")
    a, IF, tau, TM = 1e8, 50.0, 60e-9, 10e-9
    worst = 0.0
    for td in (1e-9, 20e-9, 60e-9, 200e-9, 2e-6):
        r = ramp_recovery_dead_time(a, IF, tau, TM, td)
        In, Qn = integrate_dead_time(a, IF, tau, TM, td)
        e = max(abs(In/r["Irr"] - 1), abs(Qn/r["Qrr"] - 1))
        worst = max(worst, e)
        print(f"  td = {td*1e9:7.1f} ns: Irr = {r['Irr']:7.3f} A (numerical {In:7.3f}), "
              f"Qrr = {r['Qrr']*1e9:8.2f} nC (numerical {Qn*1e9:8.2f}), T1/tau = {r['x']:.3f}")
    r_inf = lm.ramp_recovery(a, IF, tau, TM)
    r_td = ramp_recovery_from_charge(a, IF, tau, TM, tau*IF)
    e = max(abs(r_td["Irr"]/r_inf["Irr"] - 1), abs(r_td["Qrr"]/r_inf["Qrr"] - 1))
    worst = max(worst, e)
    print(f"  steady start against lm.ramp_recovery: Irr {r_inf['Irr']:.4f} / {r_td['Irr']:.4f} A, "
          f"Qrr {r_inf['Qrr']*1e9:.3f} / {r_td['Qrr']*1e9:.3f} nC")
    C0, m = fit_coss_point_charge(2.0e-9, 50.0, 180e-9, 1.0)
    print(f"  C_oss power law from C(50 V) = 2 nF and Q(50 V) = 180 nC with Vj = 1 V: C0 = {C0*1e9:.2f} nF, m = {m:.3f}; "
          f"check Q = {float(qoss(50.0, C0, 1.0, m))*1e9:.1f} nC, C = {float(coss(50.0, C0, 1.0, m))*1e9:.3f} nF")
    i_d, v = channel_injection(50.0, 1.5e-3, 1e-9, 1.3, lm.thermal_voltage(25.0))
    print(f"  channel on, 50 A through 1.5 mOhm: diode current {i_d*1e6:.3f} uA, v = {v*1e3:.1f} mV")
    print(f"largest relative deviation: {worst:.2e}")

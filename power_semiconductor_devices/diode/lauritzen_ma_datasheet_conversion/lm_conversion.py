"""Conversion between the data of a diode datasheet and the parameters of the Lauritzen-Ma diode model.
Companion module of lauritzen_ma_datasheet_conversion.tex; used by lm_conversion_calc.py.

Model (charge control with two lumped charges):

    i_dif   = (q_E - q_M)/T_M                    diffusion current into the base
    dq_M/dt = i_dif - q_M/tau                    charge stored in the base, lifetime tau
    q_E     = (tau + T_M)*I_S*(exp(v_j/(n*V_T)) - 1)     junction law (I_S: saturation current of the static curve)
    v       = v_j + R_S*i,   i = i_dif + C_j(v_j)*dv_j/dt + v_j/R_r

While the junction is forward biased the circuit imposes the current and q_E follows; the junction blocks when
q_E reaches zero, and from then on i_dif = -q_M/T_M decays with tau_rr = tau*T_M/(tau + T_M).

Notation of this module: a = rate of fall of the current (-di/dt > 0), IF = forward current before the ramp,
Irr = peak reverse recovery current (> 0), Qrr = recovered charge (integral of the reverse current), ta = Irr/a,
T1 = (IF + Irr)/a. SI units throughout (A, s, C, V, Ohm, F, H). Only numpy and scipy are needed (numpy 1.x, 2.x).
Run:  python lm_conversion.py      (self-test of the closed forms against numerical integration, a few seconds)"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw

K_BOLTZMANN, Q_ELECTRON = 1.380649e-23, 1.602176634e-19


def thermal_voltage(T_celsius):
    """V_T = k*T/q at a temperature in degrees Celsius."""
    return K_BOLTZMANN*(T_celsius + 273.15)/Q_ELECTRON


def trapezoid(y, x):
    """integral by the trapezoidal rule (same result on numpy 1.x and 2.x)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(np.sum(0.5*(y[1:] + y[:-1])*np.diff(x)))


# ====================================================================== from the model to the datasheet
def tail_time(tau, TM):
    """Time constant of the recovery tail, tau_rr = tau*TM/(tau + TM)."""
    return tau*TM/(tau + TM)


def ramp_peak_norm(lam, rho):
    """Normalised peak reverse current y = Irr/(a*tau) on a ramp that starts from the steady state.
    lam = IF/(a*tau) (length of the ramp to the current zero, in lifetimes), rho = tau_rr/tau.
    y solves y = (1 - rho)*(1 - exp(-(lam + y))); closed form with the Lambert function."""
    lam, rho = np.asarray(lam, float), np.asarray(rho, float)
    c = 1.0 - rho
    return c + np.real(lambertw(-c*np.exp(-(lam + c)), 0))


def ramp_recovery(a, IF, tau, TM):
    """Turn-off on the ramp i = IF - a*t from the steady state at IF (exact for the model, any ramp length).
    Returns a dict: Irr, ta, T1, tau_rr, Qa (triangle), Qtail, Qrr, q0 (charge stored at the current zero),
    rho, lam, x = T1/tau, F = Irr/(a*(tau - tau_rr)), s (softness of the triangle with the same charge)."""
    trr = tail_time(tau, TM)
    rho, lam = trr/tau, IF/(a*tau)
    y = float(ramp_peak_norm(lam, rho))
    Irr = a*tau*y
    ta = Irr/a
    Qa, Qtail = 0.5*Irr*ta, Irr*trr
    return dict(Irr=Irr, ta=ta, T1=(IF + Irr)/a, tau_rr=trr, Qa=Qa, Qtail=Qtail, Qrr=Qa + Qtail,
                q0=a*tau**2*(1 - math.exp(-lam)), rho=rho, lam=lam, x=(IF + Irr)/(a*tau), F=y/(1 - rho),
                s=2*trr/ta)


def ramp_exponents(a, IF, tau, TM):
    """Local exponents of the model at one operating point: d ln(Irr)/d ln(a) and d ln(Qrr)/d ln(a) at
    constant IF. The exponents with respect to IF at constant a are one minus these (the model is linear,
    so Irr and Qrr are homogeneous functions of degree one of a and IF)."""
    r = ramp_recovery(a, IF, tau, TM)
    rho, lam, y = r["rho"], r["lam"], r["Irr"]/(a*tau)
    e = lam*(1 - rho - y)
    return 1 - e/(y*(rho + y)), 1 - e/(0.5*y*y + rho*y)


def ramp_waveform(t, a, IF, tau, TM):
    """Closed-form waveforms on the ramp; t = 0 is the start of the ramp. Returns (i, q_M, q_E)."""
    t = np.asarray(t, float)
    r = ramp_recovery(a, IF, tau, TM)
    T1, trr = r["T1"], r["tau_rr"]
    tc = np.minimum(t, T1)
    i_on = IF - a*tc
    qM_on = tau*i_on + a*tau**2*(1 - np.exp(-tc/tau))
    dec = np.exp(-np.maximum(t - T1, 0.0)/trr)
    on = t <= T1
    i = np.where(on, i_on, -r["Irr"]*dec)
    qM = np.where(on, qM_on, TM*r["Irr"]*dec)
    qE = np.where(on, qM_on + TM*i_on, 0.0)
    i = np.where(t < 0, IF, i)
    qM = np.where(t < 0, tau*IF, qM)
    qE = np.where(t < 0, (tau + TM)*IF, qE)
    return i, qM, qE


def trr_factor(kind="line", pA=0.9, pB=0.25, p=0.1):
    """Factor kappa of the fall time of an exponential tail, t_b = kappa*tau_rr, for one definition of the
    reverse recovery time t_rr = ta + t_b:
      "line"   straight line through the points at pA*Irr and pB*Irr of the tail, extended to zero current;
      "point"  instant at which the current has fallen to p*Irr;
      "charge" triangle that holds the same charge as the tail (kappa = 2)."""
    if kind == "line":
        return math.log(1/pB) + pB*math.log(pA/pB)/(pA - pB)
    if kind == "point":
        return math.log(1/p)
    if kind == "charge":
        return 2.0
    raise ValueError(kind)


def step_recovery(IF, IR, tau, TM):
    """Turn-off with a constant reverse current IR imposed by the circuit (resistive test), from the steady
    state at IF. Returns (ts, I0, Q): storage time, current at the start of the tail and recovered charge.
    If IF/IR <= TM/tau the junction blocks at once, the current jumps to IF*tau/TM and there is no plateau."""
    trr = tail_time(tau, TM)
    if IF*tau <= IR*TM:
        I0 = IF*tau/TM
        return 0.0, I0, I0*trr
    ts = tau*math.log((1 + IF/IR)*tau/(tau + TM))
    return ts, IR, IR*(ts + trr)


def sine_peak(Ipk, w, tau, TM):
    """Peak reverse current after a sinusoidal current Ipk*sin(w*t) that has lasted several lifetimes: the
    junction charge is the current filtered by (1 + s*tau_rr)/(1 + s*tau), so it crosses zero later than the
    current by the phase angle phi. Returns (Irr, phi)."""
    phi = math.atan(w*tau) - math.atan(w*tail_time(tau, TM))
    return Ipk*math.sin(phi), phi


# ====================================================================== from the datasheet to the model
def lm_from_tail(a, IF, Irr, tau_rr, long_ramp=False, tol=1e-13):
    """Lifetime and transit time from the peak Irr and the time constant of the tail, at the slope a and the
    forward current IF. Exact: fixed-point iteration tau = tau_rr + ta/(1 - exp(-T1/tau)), which converges
    for every data set (contraction ratio below Irr/(IF + Irr)). long_ramp=True (or IF = None): closed form
    tau = tau_rr + ta, valid when (IF + Irr)/a is several lifetimes. Returns (tau, TM, iterations)."""
    if tau_rr <= 0:
        raise ValueError("tail time constant <= 0: the data leave no charge for the tail")
    ta = Irr/a
    tau, n = tau_rr + ta, 0
    if not long_ramp and IF is not None:
        T1 = (IF + Irr)/a
        for n in range(1, 2000):
            new = tau_rr + ta/(1 - math.exp(-T1/tau))
            done = abs(new - tau) <= tol*new
            tau = new
            if done:
                break
    return tau, tau*tau_rr/(tau - tau_rr), n


def lm_from_irr_qrr(a, IF, Irr, Qrr, long_ramp=False, tail_fraction=1.0):
    """(tau, TM, tau_rr) from the peak reverse current and the recovered charge. tail_fraction < 1 when the
    datasheet charge is integrated only down to a residual current, e.g. 0.98 for 2 % of Irr."""
    tau_rr = (Qrr/Irr - Irr/(2*a))/tail_fraction
    tau, TM, _ = lm_from_tail(a, IF, Irr, tau_rr, long_ramp)
    return tau, TM, tau_rr


def lm_from_irr_trr(a, IF, Irr, trr, kappa, long_ramp=False):
    """(tau, TM, tau_rr) from the peak reverse current and the reverse recovery time, defined with kappa."""
    tau_rr = (trr - Irr/a)/kappa
    tau, TM, _ = lm_from_tail(a, IF, Irr, tau_rr, long_ramp)
    return tau, TM, tau_rr


def irr_from_qrr_trr(a, Qrr, trr, kappa):
    """Peak reverse current that is consistent with a charge and a recovery time (exponential tail)."""
    if abs(kappa - 2) < 1e-12:
        return 2*Qrr/trr
    disc = trr**2 + 2*kappa*(kappa - 2)*Qrr/a
    if disc < 0:
        raise ValueError("charge and recovery time are not consistent with an exponential tail")
    return a*(math.sqrt(disc) - trr)/(kappa - 2)


def irr_from_qrr_softness(a, Qrr, s):
    """Peak reverse current from the charge and an assumed softness s = 2*tau_rr/ta."""
    return math.sqrt(2*a*Qrr/(1 + s))


def sensitivities(a, IF, Irr, Qrr, long_ramp=False, h=1e-6):
    """Logarithmic sensitivities of tau, TM and tau_rr to the data: rows (tau, TM, tau_rr), columns
    (Qrr, Irr, a, IF), by central differences."""
    base = np.array([Qrr, Irr, a, IF], float)

    def f(p):
        return np.log(lm_from_irr_qrr(p[2], p[3], p[1], p[0], long_ramp))
    S = np.zeros((3, 4))
    for j in range(4):
        up, dn = base.copy(), base.copy()
        up[j] *= 1 + h
        dn[j] *= 1 - h
        S[:, j] = (f(up) - f(dn))/(math.log(1 + h) - math.log(1 - h))
    return S


# ====================================================================== forward characteristic
def forward_voltage(i, IS, n, VT, RS):
    """Static characteristic of the model, v = n*VT*ln(1 + i/IS) + RS*i."""
    return n*VT*np.log1p(np.asarray(i, float)/IS) + RS*np.asarray(i, float)


def forward_tangent(VT0, rT, I0, n, VT):
    """(IS, RS) that make the model tangent to the straight line VT0 + rT*i at the current I0."""
    RS = rT - n*VT/I0
    return I0/math.expm1((VT0 + n*VT)/(n*VT)), RS


def forward_secant(VT0, rT, I1, I2, n, VT):
    """(IS, RS) that make the model cross the straight line VT0 + rT*i at the currents I1 and I2."""
    def gap(RS):
        IS = I1/math.expm1((VT0 + (rT - RS)*I1)/(n*VT))
        return float(forward_voltage(I2, IS, n, VT, RS)) - (VT0 + rT*I2)
    RS0 = rT - n*VT*math.log(I2/I1)/(I2 - I1)               # exact when IS << I1
    RS = brentq(gap, RS0 - 0.2*abs(rT), RS0 + 0.2*abs(rT), xtol=1e-18, rtol=1e-14)
    return I1/math.expm1((VT0 + (rT - RS)*I1)/(n*VT)), RS


def forward_points(points, VT, n=None):
    """Fit through points [(I, V), ...] of the forward characteristic. Two points with n given: (IS, RS).
    Three points with n = None: (IS, RS, n). Assumes IS << I, which holds for any on-state point."""
    (I1, V1), (I2, V2) = points[0], points[1]
    if n is None:
        I3, V3 = points[2]
        A = np.array([[math.log(I2/I1), I2 - I1], [math.log(I3/I2), I3 - I2]])
        nVT, RS = np.linalg.solve(A, np.array([V2 - V1, V3 - V2]))
        IS = I1*math.exp(-(V1 - RS*I1)/nVT)
        return IS, RS, nVT/VT
    RS = (V2 - V1 - n*VT*math.log(I2/I1))/(I2 - I1)
    return I1*math.exp(-(V1 - RS*I1)/(n*VT)), RS


# ====================================================================== depletion capacitance
def cj(vj, Cj0, Vj=0.7, m=0.5, FC=0.5):
    """Depletion capacitance of the junction, C = Cj0/(1 - v_j/Vj)^m, continued linearly above FC*Vj."""
    vj = np.asarray(vj, float)
    lo = Cj0/np.power(np.maximum(1 - vj/Vj, 1 - FC), m)
    hi = Cj0/(1 - FC)**(1 + m)*(1 - FC*(1 + m) + m*vj/Vj)
    return np.where(vj < FC*Vj, lo, hi)


def cj_charge(VR, Cj0, Vj=0.7, m=0.5):
    """Charge of the depletion capacitance between zero and the reverse voltage VR."""
    return Cj0*Vj/(1 - m)*((1 + VR/Vj)**(1 - m) - 1)


# ====================================================================== commutation cell (numerical check)
def simulate_cell(dev, IF, VR, L, Rsn=None, Csn=None, t_end=None, points=20001, rtol=1e-9, method="Radau"):
    """Commutation cell: the diode carries the current IF of a current source; at t = 0 a switch applies the
    voltage VR in reverse through the inductance L, so the diode current falls with about VR/L. An optional
    Rsn-Csn snubber sits across the diode.
    dev: dict with tau, TM, IS, n, VT, RS, Rr, Cj0 and optionally Vj, m.
    States: current of the switch, q_M, junction voltage, voltage of the snubber capacitor (stiff system,
    implicit integration with the analytical Jacobian).
    Returns a dict with the arrays t, i (diode), v (diode), i_dif, qM, qE, i_sn, vj."""
    tau, TM, IS, n, VT = dev["tau"], dev["TM"], dev["IS"], dev["n"], dev["VT"]
    RS, Rr, Cj0 = dev["RS"], dev.get("Rr", 1e12), dev["Cj0"]
    Vj, m, FC = dev.get("Vj", 0.7), dev.get("m", 0.5), 0.5
    snub = Rsn is not None and Csn is not None
    g = 1/(1 + RS/Rsn) if snub else 1.0
    gs = g/Rsn if snub else 0.0                              # d(i_D)/d(v_cs) = -d(i_D)/d(v_j)
    nVT, qS = n*VT, (tau + TM)*IS

    def cap(vj):                                             # depletion capacitance and its derivative
        if vj < FC*Vj:
            c = Cj0*(1 - vj/Vj)**(-m)
            return c, c*m/(Vj - vj)
        k = Cj0/(1 - FC)**(1 + m)
        return k*(1 - FC*(1 + m) + m*vj/Vj), k*m/Vj

    def parts(y):
        isw, qM, vj, vcs = y
        ex = math.exp(min(vj/nVT, 80.0))
        iD = g*(IF - isw) - gs*(vj - vcs)
        return iD, qS*(ex - 1), qS*ex/nVT, IF - isw - iD

    def rhs(t, y):
        isw, qM, vj, vcs = y
        iD, qE, _, isn = parts(y)
        c, _ = cap(vj)
        return [(VR + vj + RS*iD)/L, (qE - qM)/TM - qM/tau, (iD - (qE - qM)/TM - vj/Rr)/c,
                isn/Csn if snub else 0.0]

    def jac(t, y):
        isw, qM, vj, vcs = y
        iD, qE, dqE, isn = parts(y)
        c, dc = cap(vj)
        f3 = (iD - (qE - qM)/TM - vj/Rr)/c
        J = np.zeros((4, 4))
        J[0] = [-RS*g/L, 0.0, (1 - RS*gs)/L, RS*gs/L]
        J[1] = [0.0, -1/TM - 1/tau, dqE/TM, 0.0]
        J[2] = [-g/c, 1/(TM*c), (-gs - dqE/TM - 1/Rr)/c - f3*dc/c, gs/c]
        if snub:
            J[3] = [(g - 1)/Csn, 0.0, gs/Csn, -gs/Csn]
        return J

    vj0 = nVT*math.log1p(IF/IS)
    for _ in range(50):                                      # steady state with the leakage branch
        vj0 = nVT*math.log1p((IF - vj0/Rr)/IS)
    y0 = [0.0, tau*(IF - vj0/Rr), vj0, vj0 + RS*IF]
    r = ramp_recovery(VR/L, IF, tau, TM)
    if t_end is None:
        t_end = r["T1"] + 14*r["tau_rr"]
    t = np.linspace(0.0, t_end, points)
    sol = solve_ivp(rhs, [0.0, t_end], y0, method=method, jac=jac, t_eval=t, rtol=rtol,
                    atol=[1e-9*IF, 1e-9*tau*IF, 1e-8, 1e-8])
    if not sol.success:
        raise RuntimeError(sol.message)
    isw, qM, vj, vcs = sol.y
    iD = g*(IF - isw) - gs*(vj - vcs)
    qE = qS*np.expm1(np.minimum(vj/nVT, 80.0))
    return dict(t=sol.t, i=iD, v=vj + RS*iD, i_dif=(qE - qM)/TM, qM=qM, qE=qE, i_sn=IF - isw - iD, vj=vj,
                steps=sol.nfev)


def measure(t, i, pA=0.9, pB=0.25, p_end=0.02, window=5e-3):
    """Datasheet quantities of a recovery waveform i(t): slope a at the current zero (mean slope over the last
    `window` of the forward current before the zero), Irr, ta, Qrr integrated from the zero to the instant at
    which the current has fallen to p_end*Irr, trr by the straight line through pA*Irr and pB*Irr, and the
    instant at which the current has fallen to 10 % of Irr."""
    t, i = np.asarray(t, float), np.asarray(i, float)
    k0 = int(np.argmax(i < 0))
    t0 = t[k0 - 1] + (t[k0] - t[k0 - 1])*i[k0 - 1]/(i[k0 - 1] - i[k0])
    kp = int(np.argmin(i))
    Irr, tp = -i[kp], t[kp]
    k1 = int(np.argmax(i[:k0] < window*i[0])) if np.any(i[:k0] < window*i[0]) else max(k0 - 5, 0)
    a = -(i[k0 - 1] - i[k1])/(t[k0 - 1] - t[k1]) if k0 - 1 > k1 else (i[k0 - 1] - i[k0])/(t[k0] - t[k0 - 1])

    def t_at(frac):                                           # first instant after the peak with -i = frac*Irr
        tail = -i[kp:]
        k = int(np.argmax(tail <= frac*Irr))
        if k == 0:
            return float("nan")
        k += kp
        return t[k - 1] + (t[k] - t[k - 1])*(-i[k - 1] - frac*Irr)/(-i[k - 1] + i[k])

    tA, tB, tE, t10 = t_at(pA), t_at(pB), t_at(p_end), t_at(0.1)
    sel = (t >= t0) & (t <= tE)
    tt = np.concatenate(([t0], t[sel], [tE]))
    ii = np.concatenate(([0.0], -i[sel], [p_end*Irr]))
    return dict(a=a, t0=t0, Irr=Irr, ta=tp - t0, Qrr=trapezoid(ii, tt),
                trr_line=tB + pB*(tB - tA)/(pA - pB) - t0, trr_10=t10 - t0)


def integrate_model(a, IF, tau, TM, dt=None):
    """Independent check of the closed forms: the two charge equations integrated step by step (fourth-order
    Runge-Kutta) with the imposed ramp until q_E = 0 and with q_E = 0 afterwards. Returns (Irr, Qrr)."""
    dt = dt or min(tau, TM)/4000
    qM, t, t0 = IF*tau, 0.0, IF/a

    def f(tt, q):
        return IF - a*tt - q/tau
    while True:                                              # conducting: the circuit imposes the current
        k1 = f(t, qM)
        k2 = f(t + dt/2, qM + dt/2*k1)
        k3 = f(t + dt/2, qM + dt/2*k2)
        k4 = f(t + dt, qM + dt*k3)
        qn = qM + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if qn + TM*(IF - a*(t + dt)) <= 0:                   # q_E crosses zero inside this step: secant
            e0, e1 = qM + TM*(IF - a*t), qn + TM*(IF - a*(t + dt))
            h = dt*e0/(e0 - e1)
            qM, t = qM + (qn - qM)*h/dt, t + h
            break
        qM, t = qn, t + dt
    Irr = a*t - IF
    Q = 0.5*a*(t - t0)**2
    lam = 1/tau + 1/TM
    while qM/TM > 1e-9*Irr:                                  # blocked: q_E = 0, i = -q_M/T_M
        k1 = -lam*qM
        k2 = -lam*(qM + dt/2*k1)
        k3 = -lam*(qM + dt/2*k2)
        k4 = -lam*(qM + dt*k3)
        qn = qM + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        Q += 0.5*(qM + qn)/TM*dt
        qM = qn
    return Irr, Q


# ====================================================================== self-test
if __name__ == "__main__":
    print("Lauritzen-Ma conversion: self-test of the closed forms")
    worst = 0.0
    for a, IF, tau, TM in ((5e6, 1000.0, 38e-6, 20e-6), (1e9, 100.0, 0.28e-6, 0.18e-6), (2e8, 50.0, 1e-6, 0.1e-6),
                           (1.76e5, 100.0, 76.4e-6, 14.4e-6), (5e9, 30.0, 0.5e-6, 1.5e-6)):
        r = ramp_recovery(a, IF, tau, TM)
        In, Qn = integrate_model(a, IF, tau, TM)
        tau2, TM2, _ = lm_from_irr_qrr(a, IF, r["Irr"], r["Qrr"])
        e = max(abs(In/r["Irr"] - 1), abs(Qn/r["Qrr"] - 1), abs(tau2/tau - 1), abs(TM2/TM - 1))
        worst = max(worst, e)
        print(f"  a = {a:8.3g} A/s, IF = {IF:6.1f} A, tau = {tau*1e6:6.2f} us, TM = {TM*1e6:5.2f} us: "
              f"T1/tau = {r['x']:5.2f}, Irr = {r['Irr']:8.3f} A (numerical {In:8.3f}), "
              f"Qrr = {r['Qrr']*1e6:9.3f} uC (numerical {Qn*1e6:9.3f}), inversion error {max(abs(tau2/tau-1), abs(TM2/TM-1)):.1e}")
    print(f"largest relative deviation: {worst:.2e}")
    print("factors kappa of t_b = kappa*tau_rr: line through 90 % and 25 % = {:.4f}, point at 10 % = {:.4f}, "
          "point at 25 % = {:.4f}, equal charge = {:.1f}".format(trr_factor("line"), trr_factor("point", p=0.1),
                                                                trr_factor("point", p=0.25), trr_factor("charge")))

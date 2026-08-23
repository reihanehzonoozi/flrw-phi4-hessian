#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical companion to

    "Spatial curvature, boundary pinning and the S^3 sphaleron: the Hessian
     spectrum of phi^4 interfaces in Friedmann-Lemaitre-Robertson-Walker spacetimes"

Reproduces every number quoted in sections 3-5 and regenerates figures 1-3.

Contents
--------
    profile(k)              pinned radial configuration, Eq. (3.7)
    hessian_spectrum(k)     reduced Hessian, Eq. (4.4), Dirichlet box
    table_spectra()         Table 4
    table_angular()         angular sectors quoted in section 4.3
    table_box_law()         box law, Eq. (4.8)
    table_robustness()      Table 5
    table_cutoff()          chi_min drift of the noncompact gap (section 4.6)
    relaxation(k)           mode equation (5.2) on the benchmark backgrounds
    figures()               figures 1-3

Conventions: lambda = v = a = 1.  At this benchmark lambda*v^2 = a^-2 = 1, so
the two natural units coincide; eigenvalues are quoted in units of a^-2.  Note
that lambda*v^2 is NOT a fixed unit once lambda is varied (see wall_numerics.py).

Usage
-----
    python3 flrw_numerics.py            # tables only
    python3 flrw_numerics.py --figures  # tables and figures (writes figs/)

Requires numpy >= 1.20, scipy >= 1.7, matplotlib >= 3.4.
The integration helper below supports both NumPy 1.x (trapz) and 2.x (trapezoid).
"""

import argparse
import os

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import eigh_tridiagonal

# NumPy 2.x renamed trapz to trapezoid; retain compatibility with NumPy >= 1.20.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
LAM = 1.0                       # quartic coupling
VEV = 1.0                       # symmetry-breaking scale
M_PHI2 = 2.0 * LAM * VEV ** 2   # fluctuation mass squared of the broken vacuum

# numerical defaults used unless stated otherwise in the text
FIG_DIR = "figs/flrw"      # kept distinct from the companion paper
N_GRID = 500
L_OUT = 8.0
CHI_MIN = 1.0e-3
CHI_MAX_CLOSED = np.pi - 1.0e-3

dV = lambda p: LAM * p * (p ** 2 - VEV ** 2)          # V'(phi)
d2V = lambda p: LAM * (3.0 * p ** 2 - VEV ** 2)       # V''(phi)


def cotangent_k(k, chi):
    """ctn_k = S_k'/S_k: cot, 1/chi, coth for k = +1, 0, -1."""
    if k == 1:
        return 1.0 / np.tan(chi)
    if k == 0:
        return 1.0 / chi
    return 1.0 / np.tanh(chi)


def S_k(k, chi):
    """S_k(chi) = sin, chi, sinh."""
    if k == 1:
        return np.sin(chi)
    if k == 0:
        return chi
    return np.sinh(chi)


def outer_edge(k, L=L_OUT):
    """Radial domain is compact for k = +1."""
    return CHI_MAX_CLOSED if k == 1 else L


# ----------------------------------------------------------------------
# static profile: Eq. (3.7)
# ----------------------------------------------------------------------
def profile(k, L=L_OUT, chi_min=CHI_MIN, tol=1.0e-8, n_init=4000):
    """Solve phi'' + 2 ctn_k phi' = V'(phi) with phi(chi_min)=0, phi(L)=v.

    The collocation solver is initialized with the flat kink; the solution
    object returned supports evaluation at arbitrary chi through .sol.
    """
    b = outer_edge(k, L)
    chi = np.linspace(chi_min, b, n_init)
    guess = np.vstack([VEV * np.tanh(chi / np.sqrt(2.0)),
                       VEV / np.sqrt(2.0) / np.cosh(chi / np.sqrt(2.0)) ** 2])

    def rhs(t, y):
        return np.vstack([y[1], dV(y[0]) - 2.0 * cotangent_k(k, t) * y[1]])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - VEV])

    sol = solve_bvp(rhs, bc, chi, guess, tol=tol, max_nodes=200000)
    if sol.status != 0:
        raise RuntimeError("BVP failed for k = %+d (status %d)" % (k, sol.status))
    return sol


# ----------------------------------------------------------------------
# reduced Hessian: Eq. (4.4), plus the j-barrier of section 4.3
# ----------------------------------------------------------------------
def hessian_spectrum(k, j=0, N=N_GRID, L=L_OUT, chi_min=CHI_MIN,
                     inner_bc="dirichlet", n_levels=5, return_modes=False,
                     common_profile=None):
    """Eigenvalues of -d^2/dchi^2 + V''(phi_0) - k + j(j+1)/S_k^2.

    Second-order finite-volume/finite-difference discretization on a uniform
    grid; the matrix is real symmetric tridiagonal, so the spectrum is real.
    Dirichlet boundary nodes are eliminated exactly. For the optional inner
    Neumann condition the boundary node is retained with trapezoidal weight.
    """
    b = outer_edge(k, L)
    sol = profile(k, L=L, chi_min=chi_min)
    chi = np.linspace(chi_min, b, N)
    h = chi[1] - chi[0]

    phi = common_profile(chi) if common_profile is not None else sol.sol(chi)[0]
    W = d2V(phi) - k
    if j:
        W = W + j * (j + 1) / S_k(k, chi) ** 2

    # Finite-volume quadratic form with trapezoidal node weights.  Dirichlet
    # nodes are eliminated exactly rather than penalized, and the diagonal mass
    # matrix is removed by the symmetric map B^{-1/2} A B^{-1/2}, so the problem
    # stays tridiagonal and costs O(N) storage.
    mu = np.ones(N)
    mu[0] = mu[-1] = 0.5
    diag = 2.0 * np.ones(N) / h ** 2
    diag[0] = diag[-1] = 1.0 / h ** 2
    diag = diag + mu * W
    off = -np.ones(N - 1) / h ** 2

    keep = np.ones(N, dtype=bool)
    keep[-1] = False                       # outer Dirichlet (box truncation)
    if inner_bc == "dirichlet":
        keep[0] = False
    elif inner_bc != "neumann":
        raise ValueError("inner_bc must be 'dirichlet' or 'neumann'")

    d = diag[keep] / mu[keep]
    idx = np.flatnonzero(keep)
    e = off[idx[:-1]] / np.sqrt(mu[idx[:-1]] * mu[idx[1:]])

    if return_modes:
        w, U = eigh_tridiagonal(d, e, select='i', select_range=(0, n_levels - 1))
        u = np.zeros(N)
        u[keep] = U[:, 0] / np.sqrt(mu[keep])
        u = u * np.sign(np.sum(u))          # fix the arbitrary overall sign
        return w, u, chi
    return eigh_tridiagonal(d, e, eigvals_only=True,
                            select='i', select_range=(0, n_levels - 1))


def morse_index(levels, tol=1.0e-8):
    return int(np.sum(levels < -tol))


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------
def table_spectra():
    """Table 4: lowest levels, Morse index, offset from the flat case."""
    print("\nTable 4  (N=%d, L=%g, chi_min=%g, lambda=v=a=1)" % (N_GRID, L_OUT, CHI_MIN))
    print("  section   k    lam_min      lam_2      ind    lam_min(k)-lam_min(0)")
    flat = hessian_spectrum(0)[0]
    for k, name in ((1, "S^3"), (0, "R^3"), (-1, "H^3")):
        w = hessian_spectrum(k)
        print("  %-8s %+d   %+8.3f   %+8.3f    %d      %+8.3f"
               % (name, k, w[0], w[1], morse_index(w), w[0] - flat))


def table_angular():
    """Lowest level in the sectors j = 0, 1, 2 (section 4.3)."""
    print("\nAngular sectors, lowest level (section 4.3)")
    for k, name in ((0, "R^3"), (-1, "H^3"), (1, "S^3")):
        vals = [hessian_spectrum(k, j=j)[0] for j in (0, 1, 2)]
        print("  %-4s  j=0,1,2 : %s" % (name, ", ".join("%.3f" % x for x in vals)))
    print("  (S^3 follows the exact ladder 1 + (j+n)^2 of the trigonometric")
    print("   Poeschl-Teller problem; R^3 follows the spherical-Bessel box zeros)")


def table_convergence():
    """Grid convergence of the lowest s-wave level used in Table 4."""
    print("\nGrid convergence of lambda_min (L=%g, chi_min=%g)" % (L_OUT, CHI_MIN))
    print("  N       S^3          R^3          H^3")
    for N in (200, 500, 1000, 2000, 4000):
        vals = [hessian_spectrum(k, N=N)[0] for k in (1, 0, -1)]
        print(" %4d   %10.7f   %10.7f   %10.7f" % (N, vals[0], vals[1], vals[2]))


def table_mode_overlaps():
    """Overlap of the computed ground mode with the corresponding Dirichlet box sine."""
    print("\nGround-mode overlap with the Dirichlet box sine")
    print("  section      overlap")
    for k, name in ((1, "S^3"), (0, "R^3"), (-1, "H^3")):
        _, u, chi = hessian_spectrum(k, N=4000, return_modes=True)
        edge = outer_edge(k)
        box = np.sin(np.pi * (chi - CHI_MIN) / (edge - CHI_MIN))
        num = abs(_trapz(u * box, chi))
        den = np.sqrt(_trapz(u * u, chi) * _trapz(box * box, chi))
        print("  %-4s       %.8f" % (name, num / den))


def full_robustness_scan():
    """Reproduce the parameter-range sensitivity statement in section 4.6.

    The scan varies one parameter at a time about the benchmark point.  It
    checks the two qualitative claims used in the manuscript: positivity of
    the lowest level and the ordering S^3 < R^3 < H^3.
    """
    global LAM, VEV, M_PHI2
    baseline = (LAM, VEV, M_PHI2)
    failures = []
    nrun = 0

    def check(label, **kw):
        nonlocal nrun
        vals = {k: hessian_spectrum(k, n_levels=1, **kw)[0] for k in (1, 0, -1)}
        nrun += 1
        ok = min(vals.values()) > 0.0 and vals[1] < vals[0] < vals[-1]
        if not ok:
            failures.append((label, vals))

    try:
        # Physical parameters, varied one at a time.
        for lam in (0.5, 1.0, 2.0):
            LAM = lam; VEV = 1.0; M_PHI2 = 2.0 * LAM * VEV ** 2
            check("lambda=%g" % lam, N=1000)
        for v in (0.5, 1.0, 2.0):
            LAM = 1.0; VEV = v; M_PHI2 = 2.0 * LAM * VEV ** 2
            check("v=%g" % v, N=1000)

        # Restore benchmark model before numerical-parameter scans.
        LAM, VEV, M_PHI2 = baseline
        for N in (200, 500, 1000, 2000):
            check("N=%d" % N, N=N)
        for L in (5.0, 8.0, 12.0, 20.0, 50.0):
            check("L=%g" % L, N=1000, L=L)
        for cm in (1e-3, 1e-2, 5e-2, 0.2, 1.0):
            check("chi_min=%g" % cm, N=1000, chi_min=cm)
    finally:
        LAM, VEV, M_PHI2 = baseline

    print("\nFull one-at-a-time robustness scan: %d parameter settings" % nrun)
    if failures:
        print("  FAIL: %d settings violated positivity or S^3<R^3<H^3" % len(failures))
        for label, vals in failures:
            print("   ", label, vals)
    else:
        print("  PASS: positivity and S^3 < R^3 < H^3 held in every setting")


def table_box_law():
    """Box law, Eq. (4.8): lam_n = 2 lam v^2 - k/a^2 + n^2 pi^2 / l_box^2."""
    print("\nBox law, Eq. (4.8): measured vs predicted")
    for k, name in ((0, "R^3"), (-1, "H^3"), (1, "S^3")):
        b = outer_edge(k)
        l_box = (np.pi - 2 * CHI_MIN) if k == 1 else (b - CHI_MIN)
        meas = hessian_spectrum(k, n_levels=3)
        pred = [M_PHI2 - k + n ** 2 * np.pi ** 2 / l_box ** 2 for n in (1, 2, 3)]
        print("  %-4s  measured %s" % (name, ", ".join("%.4f" % x for x in meas)))
        print("        predicted %s" % ", ".join("%.4f" % x for x in pred))


def table_robustness(N=4000):
    """Table 5: dependence on boundary prescription, cutoff and domain.

    The last column recomputes the noncompact difference with a single
    common profile, which is the hypothesis under which Corollary 1 holds
    exactly; the difference between the two gap columns is the profile
    contribution.
    """
    print("\nTable 5  (N=%d, lambda=v=a=1)" % N)
    print("  configuration                        S^3      R^3      H^3     gap(self)  gap(common)")
    runs = [("Dirichlet, chi_min=1e-3, L=8", dict(chi_min=1e-3, L=8.0, inner_bc="dirichlet")),
            ("Dirichlet, chi_min=0.05, L=8", dict(chi_min=0.05, L=8.0, inner_bc="dirichlet")),
            ("Dirichlet, chi_min=1e-3, L=12", dict(chi_min=1e-3, L=12.0, inner_bc="dirichlet")),
            ("Neumann (inner), L=8", dict(chi_min=1e-3, L=8.0, inner_bc="neumann"))]
    for label, kw in runs:
        vals = {k: hessian_spectrum(k, N=N, **kw)[0] for k in (1, 0, -1)}
        flat = profile(0, L=kw["L"], chi_min=kw["chi_min"])
        common = lambda x: flat.sol(x)[0]
        gc = (hessian_spectrum(-1, N=N, common_profile=common, **kw)[0]
              - hessian_spectrum(0, N=N, common_profile=common, **kw)[0])
        print("  %-34s %+.3f   %+.3f   %+.3f    %.5f   %.8f"
               % (label, vals[1], vals[0], vals[-1], vals[-1] - vals[0], gc))


def table_box_deficit():
    """Why the box law degrades once the inner cutoff is coarsened."""
    print("\nDeparture from the box law at a coarse cutoff (chi_min = 0.05)")
    for k, name in ((1, "S^3"), (0, "R^3"), (-1, "H^3")):
        b = outer_edge(k)
        l_box = (b - 0.05)
        pred = M_PHI2 - k + np.pi ** 2 / l_box ** 2
        meas = hessian_spectrum(k, N=4000, chi_min=0.05)[0]
        sol = profile(k, chi_min=0.05)
        x = np.linspace(0.05, b, 40000)
        chi95 = x[np.argmax(sol.sol(x)[0] > 0.95 * VEV)]
        print("  %-4s  box %.4f   measured %.4f   deficit %.4f   phi_0=0.95v at chi=%.3f"
               % (name, pred, meas, pred - meas, chi95))


def table_layer_collapse(k=0, L=L_OUT):
    """Energy and boundary slope of the pinned layer as the cutoff is removed.

    The near-origin solution of (S_k^2 phi')' = 0 is phi = A - C/chi with
    C = v chi_min, so phi'(chi_min) = v/chi_min and the gradient energy is
    2 pi v^2 chi_min: the layer collapses onto the vacuum as chi_min -> 0.
    """
    print("\nCollapse of the pinned layer as the cutoff is removed (k = %+d, L = %g)" % (k, L))
    print("  chi_min     energy      2 pi v^2 chi_min   phi'(chi_min)   chi_min*phi'")
    for cm in (1e-1, 5e-2, 1e-2, 3e-3, 1e-3):
        sol = profile(k, L=L, chi_min=cm)
        x = np.linspace(cm, outer_edge(k, L), 200000)
        p, dp = sol.sol(x)
        e = 4.0 * np.pi * _trapz(S_k(k, x) ** 2 * (0.5 * dp ** 2 + 0.25 * LAM *
                                       (p ** 2 - VEV ** 2) ** 2), x)
        print("  %7.0e   %9.5f   %12.5f   %13.2f   %10.4f"
               % (cm, e, 2 * np.pi * VEV ** 2 * cm, dp[0], cm * dp[0]))


def table_cutoff():
    """Drift of the noncompact gap with the inner cutoff (section 4.6)."""
    print("\nInner-cutoff dependence of the noncompact gap (section 4.6)")
    for chi_min in (1e-3, 1e-2, 5e-2, 0.2, 1.0):
        a = hessian_spectrum(0, chi_min=chi_min)[0]
        b = hessian_spectrum(-1, chi_min=chi_min)[0]
        print("  chi_min = %5.3f :  R^3 = %.4f   H^3 = %.4f   gap = %.5f"
               % (chi_min, a, b, b - a))


# ----------------------------------------------------------------------
# reduced damping model: Eq. (5.2) on the benchmarks of Table 6
# ----------------------------------------------------------------------
SCALE_FACTOR = {1: np.sin, 0: lambda t: t ** (2.0 / 3.0), -1: np.sinh}
HUBBLE = {1: lambda t: 1.0 / np.tan(t),
          0: lambda t: 2.0 / (3.0 * t),
          -1: lambda t: 1.0 / np.tanh(t)}


def relaxation(k, lam_min, t0=0.5, t1=10.0, c0=1.0e-2, rtol=1.0e-8, atol=1.0e-10):
    """Integrate c'' + 3 H(t) c' + lam_min c = 0 with c(t0)=c0, c'(t0)=0."""
    if k == 1:
        t1 = min(t1, np.pi / 2.0)          # expansion phase only

    def rhs(t, y):
        return [y[1], -3.0 * HUBBLE[k](t) * y[1] - lam_min * y[0]]

    return solve_ivp(rhs, (t0, t1), [c0, 0.0], rtol=rtol, atol=atol,
                     dense_output=True, max_step=0.01)


HDOT = {1: lambda t: -1.0 / np.sin(t) ** 2,
        0: lambda t: -2.0 / (3.0 * t ** 2),
        -1: lambda t: -1.0 / np.sinh(t) ** 2}


def wkb_frequency(k, lam, t):
    """q(t) = lam - (9/4) H^2 - (3/2) Hdot, the exact coefficient obtained from
    c = a^{-3/2} y.  The Hdot term is not negligible: for a = t^{2/3} it cancels
    the H^2 term identically.

    This same q is the squared WKB frequency of Eq. (5.7): the adiabatic form
    is c(t) ~ A cos[int omega dt + phi] / (a^{3/2} sqrt(omega)), with
    omega^2 = q.  The geometric factor a^{-3/2} is the dominant part of the
    envelope, not the whole of it in general, but on both benchmarks used here
    the adiabatic factor omega^{-1/2} is constant or asymptotically constant:
    for a = t^{2/3} one has q = lam identically, and for a = sinh(t) one has
    H -> 1 and Hdot -> 0, so q -> lam - 9/4."""
    return lam - 2.25 * HUBBLE[k](t) ** 2 - 1.5 * HDOT[k](t)


def table_dynamics():
    print("\nRelaxation of the lowest mode (initial amplitude 1e-2 at t0=0.5)")
    lmin = {k: hessian_spectrum(k)[0] for k in (1, 0, -1)}
    Hc = (2.0 / 3.0) * np.sqrt(lmin[-1])
    print("  constant-H estimate  H_c = (2/3) sqrt(lam_min[H^3]) = %.3f" % Hc)
    print("  q(t) = lam - (9/4)H^2 - (3/2)Hdot :")
    tt = np.linspace(0.05, 5.0, 200000)
    q0 = wkb_frequency(0, lmin[0], tt)
    print("     flat benchmark a=t^(2/3): max|q - lam| = %.2e  (exact cancellation)"
           % np.max(np.abs(q0 - lmin[0])))
    qm = wkb_frequency(-1, lmin[-1], tt)
    i = np.flatnonzero(np.diff(np.sign(qm)))[0]
    print("     a=sinh(t): sign change of q at t_* = %.3f  (not %.2f from the "
          "constant-H criterion)" % (tt[i], np.arctanh(1.0 / Hc)))
    for k, name, times in ((0, "R^3", (5.0, 20.0)), (-1, "H^3", (5.0, 10.0, 20.0, 50.0))):
        sol = relaxation(k, lmin[k], t1=max(times))
        vals = ", ".join("|c|(%g) = %.2e" % (t, abs(sol.sol(t)[0])) for t in times)
        print("  %-4s %s" % (name, vals))


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def figures(outdir=FIG_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "figure.dpi": 300, "savefig.bbox": "tight"})
    # set usetex = True below to typeset figure text with the manuscript fonts
    # plt.rcParams["text.usetex"] = True

    col = {1: "#D55E00", 0: "#009E73", -1: "#0072B2"}      # color-blind safe
    dash = {1: "-", 0: "--", -1: "-."}
    lab = {1: r"$k=+1$ ($S^3$)", 0: r"$k=0$ ($\mathbb{R}^3$)", -1: r"$k=-1$ ($H^3$)"}

    # ---- figure 1: profiles and effective potentials
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for k in (1, 0, -1):
        sol = profile(k)
        x = np.linspace(CHI_MIN, outer_edge(k), 1500)
        ax[0].plot(x, sol.sol(x)[0], dash[k], color=col[k], lw=1.8, label=lab[k])
        ax[1].plot(x, d2V(sol.sol(x)[0]) - k, dash[k], color=col[k], lw=1.8, label=lab[k])
        ax[1].axhline(M_PHI2 - k, color=col[k], lw=0.7, ls=":", alpha=0.8)
    ax[0].set_xlabel(r"$\chi$")
    ax[0].set_ylabel(r"$\phi_0(\chi)/v$")
    ax[0].set_title("(a) pinned radial profiles", fontsize=11)
    ax[0].set_xlim(0, 8)
    ax[0].set_ylim(-0.03, 1.08)
    ax[0].legend(frameon=False)
    ins = ax[0].inset_axes([0.42, 0.18, 0.5, 0.45])
    xi = np.linspace(CHI_MIN, 0.06, 400)
    for k in (1, 0, -1):
        ins.plot(xi, profile(k).sol(xi)[0], dash[k], color=col[k], lw=1.4)
    ins.plot(xi, np.tanh(xi / np.sqrt(2.0)), color="0.35", lw=1.2, ls=(0, (1, 1)))
    ins.text(0.028, 0.18, "planar kink", fontsize=8, color="0.35")
    ins.text(0.005, 0.78, "pinning layer", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.set_xlim(0, 0.06)
    ax[1].set_xlabel(r"$\chi$")
    ax[1].set_ylabel(r"$W_k(\chi)=V''(\phi_0)-k/a^2$")
    ax[1].set_title("(b) effective potentials", fontsize=11)
    ax[1].annotate("", xy=(6.9, 3), xytext=(6.9, 1), arrowprops=dict(arrowstyle="<->", lw=1))
    ax[1].text(6.75, 2.0, r"offsets $-k$", fontsize=9, ha="right", va="center",
               bbox=dict(fc="w", ec="none", pad=0.5))
    ax[1].set_xlim(0, 8)
    ax[1].set_ylim(-2.2, 3.6)
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(outdir, "fig1_profiles.pdf"))
    plt.close(fig)

    # ---- figure 2: spectra and ground modes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    slot = {1: 0, 0: 1, -1: 2}
    for k in (1, 0, -1):
        w, u, chi = hessian_spectrum(k, return_modes=True)
        for n, e in enumerate(w):
            ax[0].hlines(e, slot[k] - 0.28, slot[k] + 0.28,
                         color=col[k], lw=(2.6 if n == 0 else 1.2))
        ax[0].hlines(M_PHI2 - k, -0.45, 2.45, color="0.55", lw=0.8, ls=":")
        ax[0].text(2.62, M_PHI2 - k + 0.13, r"$2\lambda v^2-k/a^2=%d$" % (M_PHI2 - k),
                   fontsize=8.5, color="0.4", va="bottom")
        ax[0].text(slot[k], w[0] + (0.30 if k else -0.62),
                   r"$\lambda_{\min}=%.3f$" % w[0], ha="center", fontsize=9,
                   color=col[k], bbox=dict(fc="w", ec="none", pad=0.6))
        ax[1].plot(chi, u / np.max(np.abs(u)), dash[k], color=col[k], lw=1.8, label=lab[k])
        edge = outer_edge(k)
        ax[1].plot(chi, np.sin(np.pi * (chi - CHI_MIN) / (edge - CHI_MIN)),
                   color="0.35", lw=0.9, ls=(0, (1, 1.5)))
    ax[0].annotate("", xy=(2.30, 3.154), xytext=(2.30, 2.154),
                   arrowprops=dict(arrowstyle="<->", lw=1))
    ax[0].text(2.36, 2.65, r"$-k=1.000$", fontsize=9, rotation=90, va="center")
    ax[0].hlines(2.154, 1.28, 2.30, color="0.5", lw=0.6)
    ax[0].hlines(3.154, 2.28, 2.30, color="0.5", lw=0.6)
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels([r"$S^3$", r"$\mathbb{R}^3$", r"$H^3$"])
    ax[0].set_ylabel(r"eigenvalues $\lambda_n\ [\lambda v^2]$")
    ax[0].set_ylim(0, 11)
    ax[0].set_xlim(-0.5, 3.9)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_title(r"(a) low-lying s-wave spectrum; dotted: thresholds", fontsize=10)
    ax[1].set_xlabel(r"$\chi$")
    ax[1].set_ylabel(r"$\eta_0(\chi)$ (normalized)")
    ax[1].set_title("(b) ground modes vs Dirichlet box modes (dotted)", fontsize=10)
    # pin both axes to the origin so the two zeros meet at the corner
    ax[1].set_xlim(0, 8)
    ax[1].set_ylim(0, 1.05)
    ax[1].spines["left"].set_position(("data", 0.0))
    ax[1].spines["bottom"].set_position(("data", 0.0))
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(outdir, "fig2_spectrum.pdf"))
    plt.close(fig)

    # ---- figure 3: relaxation
    lmin = {k: hessian_spectrum(k)[0] for k in (1, 0, -1)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for k in (1, 0, -1):
        sol = relaxation(k, lmin[k])
        t = np.linspace(0.5, np.pi / 2 if k == 1 else 10.0, 4000)
        ax[0].semilogy(t, np.abs(sol.sol(t)[0]) + 1e-40, dash[k],
                       color=col[k], lw=1.5, label=lab[k])
    t = np.linspace(0.5, 10.0, 500)
    ax[0].semilogy(t, 1e-2 * (t / 0.5) ** -1, color="0.45", lw=0.9, ls=(0, (1, 1.5)))
    ax[0].semilogy(t, 1e-2 * (np.sinh(t) / np.sinh(0.5)) ** -1.5,
                   color="0.45", lw=0.9, ls=(0, (1, 1.5)))
    ax[0].text(6.2, 4e-4, r"$\propto a^{-3/2}=t^{-1}$", fontsize=9, color="0.35")
    ax[0].text(4.0, 3e-8, r"$\propto(\sinh t)^{-3/2}$", fontsize=9, color="0.35")
    ax[0].axvline(np.pi / 2, color="0.6", lw=0.8, ls=":")
    ax[0].text(np.pi / 2 + 0.08, 2e-9, r"$t=\pi/2$ turnaround ($k=+1$)",
               fontsize=8, rotation=90, color="0.35")
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel(r"$|c_1(t)|$")
    ax[0].set_ylim(1e-10, 3e-2)
    ax[0].set_title("(a) relaxation of the lowest mode", fontsize=11)
    ax[0].legend(frameon=False, loc="lower left")

    sol = relaxation(-1, lmin[-1], t1=50.0, rtol=1e-10, atol=1e-40)
    t = np.linspace(0.5, 50.0, 20000)
    ax[1].semilogy(t, np.abs(sol.sol(t)[0]) + 1e-45, "-", color=col[-1], lw=1.0)
    ax[1].semilogy(t, 1e-2 * (np.sinh(t) / np.sinh(0.5)) ** -1.5,
                   color="0.35", lw=1.0, ls=(0, (1, 1.5)))
    tt = np.linspace(0.05, 5.0, 200000)
    qm = wkb_frequency(-1, lmin[-1], tt)
    ts = tt[np.flatnonzero(np.diff(np.sign(qm)))[0]]
    ax[1].axvline(ts, color="0.6", lw=0.8, ls=":")
    ax[1].text(ts + 0.6, 1e-30, r"$q(t_*)=0$ at $t_*=%.2f$" % ts,
               fontsize=9, rotation=90, color="0.35")
    ax[1].text(24, 1e-12, r"envelope $\propto e^{-3t/2}$", fontsize=9, color="0.35")
    ax[1].set_xlabel("$t$")
    ax[1].set_ylabel(r"$|c_1(t)|$")
    ax[1].set_ylim(1e-40, 3e-2)
    ax[1].set_title("(b) hyperbolic case, long-time decay", fontsize=11)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(outdir, "fig3_dynamics.pdf"))
    plt.close(fig)
    print("\nfigures written to %s/" % outdir)


# ----------------------------------------------------------------------
def bogomolny_check():
    """Flat-space consistency check quoted in section 3.2."""
    x = np.linspace(-20.0, 20.0, 200001)
    phi = VEV * np.tanh(np.sqrt(LAM) * VEV * x / np.sqrt(2.0))
    dphi = np.gradient(phi, x)
    energy = _trapz(0.5 * dphi ** 2 + 0.25 * LAM * (phi ** 2 - VEV ** 2) ** 2, x)
    exact = 2.0 * np.sqrt(2.0) / 3.0 * np.sqrt(LAM) * VEV ** 3
    print("\nBogomolny bound: numerical %.6f, exact %.6f, relative error %.1e"
           % (energy, exact, abs(energy - exact) / exact))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="also regenerate figures 1-3")
    ap.add_argument("--full-validation", action="store_true",
                    help="also run convergence, overlap and full robustness scans")
    args, _ = ap.parse_known_args()


    bogomolny_check()
    table_spectra()
    table_angular()
    table_box_law()
    table_robustness()
    table_box_deficit()
    table_cutoff()
    table_layer_collapse()
    table_dynamics()
    if args.full_validation:
        table_convergence()
        table_mode_overlaps()
        full_robustness_scan()
    if args.figures:
        figures()


if __name__ == "__main__":
    main()
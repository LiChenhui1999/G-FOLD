# 3D convex powered-descent guidance in R^3 with tf optimization.
# Outer loop: golden-section search on tf.
# Inner loop: DPP-parameterized CVXPY problem with CLARABEL warm-start.
#
# ---- Warm-start scope (what actually speeds up, what doesn't) ----
# Two layers of reuse are active between golden-section iterations:
#
#   1. DPP parametrization (CVXPY layer): all tf-dependent terms are
#      cp.Parameter, so CVXPY canonicalizes the problem ONCE. Subsequent
#      solve() calls skip the DCP-to-cone reduction and only patch numeric
#      values into the cached A/b/P/q matrices.
#
#   2. warm_start=True (CLARABEL layer): CVXPY calls Clarabel's
#      _solver.update(P, q, A, b, settings) instead of constructing a new
#      solver object. This saves memory allocation and the sparse symbolic
#      factorization setup.
#
# What warm_start=True does NOT do with CLARABEL: it does not seed the
# solver's iterates with the previous primal/dual solution. CLARABEL is an
# interior-point method and its Python bindings via CVXPY don't expose an
# iterate-level warm-start API — each solve starts from CLARABEL's default
# central-path initializer regardless of prior results.
#
# Consequence for timing: `problem.solver_stats.solve_time` reports only
# CLARABEL's internal IPM iterations and does NOT reflect either savings
# above. To observe the real speedup, measure total wall-clock of
# problem.solve() (CVXPY overhead + solver setup + solve).
#
# MATLAB reference: GFOLD.m (fminbnd on tf, fixed N).
# Convex subproblem reference: p4_3d_tf_fixed.py in this repo.

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers '3d' projection

# ---- Vehicle fixed parameters ----
m_dry        = 1505             # Vehicle dry mass [kg]
Isp          = 225              # Specific impulse [s]
g0           = 9.80665          # Standard gravity [m/s^2]
g            = np.array([0.0, 0.0, -3.7114])  # Mars gravity, e3 vertical [m/s^2]
max_throttle = 0.8              # Max open throttle [.]
min_throttle = 0.3              # Min open throttle [.]
T_max        = 6 * 3100         # Max total thrust at 1.0 throttle [N]
phi          = 27               # Thruster cant angle [deg]

# ---- Initial conditions ----
m_wet = 1905                            # Vehicle mass with fuel [kg]
r0    = np.array([1000.0, 500.0, 1500.0])  # Initial position [m]
v0    = np.array([  20.0,  50.0,  -75.0])  # Initial velocity [m/s]

# ---- Target conditions ----
rf = np.zeros(3)
vf = np.zeros(3)

# ---- Derived constants ----
# z === ln m,  u === T/m,  s === G/m  where |T| <= G
alpha = 1.0 / (Isp * g0 * np.cos(np.deg2rad(phi)))
r1    = min_throttle * T_max * np.cos(np.deg2rad(phi))  # lower thrust bound [N]
r2    = max_throttle * T_max * np.cos(np.deg2rad(phi))  # upper thrust bound [N]

# ---- Non-dimensionalization scales ----
L_sc = np.linalg.norm(r0)               # characteristic length   [m]
V_sc = np.linalg.norm(v0)               # characteristic velocity [m/s]
A_sc = r2 / m_wet                       # characteristic acceleration [m/s^2]
g_nd = g / A_sc                         # non-dim gravity

# ---- Fixed discretization and path-constraint params ----
N     = 76
theta = 50  # thrust pointing limit from vertical [deg]
slope = 4   # glide-slope [deg]
h_gs  = 1   # glide-slope offset [m]


def build_problem(N):
    """Construct the DPP-parameterized convex subproblem once.

    All tf-dependent quantities enter as cp.Parameter so CVXPY caches the
    canonical form and CLARABEL can warm-start across golden-section iterations.
    An auxiliary variable w = z - z0_vec is introduced to keep the quadratic
    lower-bound constraint DPP-compliant (cp.square of param+var would expand
    to param*param, violating DPP).
    """
    # --- Variables (non-dimensional) ---
    r = cp.Variable((3, N))
    v = cp.Variable((3, N))
    u = cp.Variable((3, N))
    z = cp.Variable(N)
    s = cp.Variable(N)
    w = cp.Variable(N)  # auxiliary: w[i] == z[i] - z0_vec[i]

    # --- tf-dependent scalar parameters (dynamics coefficients) ---
    p_v_g = cp.Parameter()   # dt * A_sc / V_sc
    p_v_u = cp.Parameter()   # dt * A_sc / (2*V_sc)
    p_r_v = cp.Parameter()   # dt * V_sc / (2*L_sc)
    p_r_u = cp.Parameter()   # dt**2 * A_sc / (12*L_sc)
    p_z_s = cp.Parameter()   # alpha * A_sc * dt / 2

    # --- tf-dependent vector parameters (per-node mass/thrust bounds) ---
    z0_vec         = cp.Parameter(N)
    z1_vec         = cp.Parameter(N)
    mu1_over_A_vec = cp.Parameter(N, nonneg=True)
    mu2_over_A_vec = cp.Parameter(N, nonneg=True)

    cons = []

    # --- Boundary conditions (tf-independent) ---
    cons += [
        r[:, 0]     == r0 / L_sc,
        v[:, 0]     == v0 / V_sc,
        z[0]        == np.log(m_wet),
        r[:, N - 1] == rf / L_sc,
        v[:, N - 1] == vf / V_sc,
    ]

    # --- Dynamics (trapezoidal, non-dim) ---
    for i in range(N - 1):
        cons += [
            v[:, i + 1] == v[:, i] + p_v_g * g_nd + p_v_u * (u[:, i] + u[:, i + 1]),
            r[:, i + 1] == r[:, i] + p_r_v * (v[:, i] + v[:, i + 1])
                           + p_r_u * (u[:, i + 1] - u[:, i]),
            z[i + 1]    == z[i] - p_z_s * (s[i] + s[i + 1]),
        ]

    # --- Auxiliary equality so quadratic bound stays DPP ---
    cons += [w == z - z0_vec]

    # --- Thrust magnitude and mass-flow bounds ---
    for i in range(N):
        cons += [
            cp.norm(u[:, i]) <= s[i],
            s[i] >= mu1_over_A_vec[i] * (1 - w[i] + 0.5 * cp.square(w[i])),
            s[i] <= mu2_over_A_vec[i] * (1 - w[i]),
            z[i] >= z0_vec[i],
            z[i] <= z1_vec[i],
        ]

    # --- Thrust pointing constraint ---
    cons += [u[2, :] >= s * np.cos(np.deg2rad(theta))]

    # --- No sub-surface flight ---
    cons += [r[2, :] >= -1.0 / L_sc]

    # --- Glide-slope cone ---
    for i in range(N):
        cons += [
            cp.norm(r[0:2, i]) <= (r[2, i] + h_gs / L_sc) / np.tan(np.deg2rad(slope)),
        ]

    objective = cp.Maximize(z[N - 1])
    problem = cp.Problem(objective, cons)
    assert problem.is_dcp(dpp=True), "Problem is not DPP-compliant"

    vars_dict = {"r": r, "v": v, "u": u, "z": z, "s": s, "w": w}
    params_dict = {
        "p_v_g": p_v_g, "p_v_u": p_v_u,
        "p_r_v": p_r_v, "p_r_u": p_r_u,
        "p_z_s": p_z_s,
        "z0_vec": z0_vec, "z1_vec": z1_vec,
        "mu1_over_A_vec": mu1_over_A_vec,
        "mu2_over_A_vec": mu2_over_A_vec,
    }
    return problem, vars_dict, params_dict


def update_parameters(params, tf, N):
    """Refill parameter values for a new tf. Returns False if infeasible a priori."""
    dt = tf / (N - 1)

    i_arr   = np.arange(N)
    z0_term = m_wet - alpha * r2 * i_arr * dt
    z1_term = m_wet - alpha * r1 * i_arr * dt
    if np.any(z0_term <= 0.0) or np.any(z1_term <= 0.0):
        return False

    params["p_v_g"].value = dt * A_sc / V_sc
    params["p_v_u"].value = dt * A_sc / (2 * V_sc)
    params["p_r_v"].value = dt * V_sc / (2 * L_sc)
    params["p_r_u"].value = dt**2 * A_sc / (12 * L_sc)
    params["p_z_s"].value = alpha * A_sc * dt / 2

    params["z0_vec"].value         = np.log(z0_term)
    params["z1_vec"].value         = np.log(z1_term)
    params["mu1_over_A_vec"].value = (r1 / z0_term) / A_sc
    params["mu2_over_A_vec"].value = (r2 / z0_term) / A_sc
    return True


_stats = {"calls": 0, "times": []}


def solve_at(problem, vars_dict, params, tf):
    """Solve the convex subproblem at a given tf. Returns fuel_used or +inf."""
    if not update_parameters(params, tf, N):
        return np.inf
    try:
        problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    except cp.error.SolverError:
        return np.inf

    _stats["calls"] += 1
    st = problem.solver_stats
    if st is not None and st.solve_time is not None:
        _stats["times"].append(st.solve_time)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return np.inf
    z_val = vars_dict["z"].value
    if z_val is None or not np.isfinite(z_val[N - 1]):
        return np.inf
    return m_wet - np.exp(z_val[N - 1])


def golden_section(f, a, b, tol=0.5):
    """Unimodal minimization on [a, b] with golden-section search."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    x1 = b - inv_phi * (b - a)
    x2 = a + inv_phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - inv_phi * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + inv_phi * (b - a)
            f2 = f(x2)
    return 0.5 * (a + b)


# ---- tf bounds (from GFOLD.m) ----
tf_min = m_dry * np.linalg.norm(vf - v0) / r2
tf_max = (m_wet - m_dry) / (alpha * r1)
print(f"tf search interval: [{tf_min:.3f}, {tf_max:.3f}] s")

# ---- Build once ----
problem, vars_dict, params = build_problem(N)

# ---- Outer golden-section search ----
tf_opt = golden_section(
    lambda tf: solve_at(problem, vars_dict, params, tf),
    tf_min, tf_max, tol=0.5,
)

# ---- Final re-solve at tf_opt (warm-started from last iterate) ----
cost_opt = solve_at(problem, vars_dict, params, tf_opt)
if not np.isfinite(cost_opt):
    raise RuntimeError("Final solve at tf_opt returned infeasible.")

print(f"\ntf_opt     : {tf_opt:.3f} s")
print(f"Status     : {problem.status}")
print(f"Final z(N) : {vars_dict['z'].value[N-1]:.6f}")
print(f"Fuel used  : {cost_opt:.2f} kg")
print(f"Solver calls: {_stats['calls']}")
if _stats["times"]:
    print(f"Solve time - first: {_stats['times'][0]*1e3:.1f} ms, "
          f"median of rest: {np.median(_stats['times'][1:])*1e3:.1f} ms")

# ---- Post-processing: restore physical units ----
r_vals  = vars_dict["r"].value * L_sc
v_vals  = vars_dict["v"].value * V_sc
u_vals  = vars_dict["u"].value * A_sc
z_vals  = vars_dict["z"].value

m_vals   = np.exp(z_vals)
u_norms  = np.linalg.norm(u_vals, axis=0)
u_dirs   = np.rad2deg(np.arctan2(
               np.sqrt(u_vals[0, :]**2 + u_vals[1, :]**2),
               u_vals[2, :]))
T_vals   = u_norms * m_vals

tv = np.linspace(0.0, tf_opt, N)

# ---- Figure 1: Trajectory (3D) ----
fig1 = plt.figure("Trajectory")
ax   = fig1.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

h_path    = ax.plot(r_vals[0, :], r_vals[1, :], r_vals[2, :],
                    'b-', linewidth=1.5, label='Path')[0]
h_thrust  = ax.quiver(r_vals[0, :], r_vals[1, :], r_vals[2, :],
                      u_vals[0, :], u_vals[1, :], u_vals[2, :],
                      length=25.0, normalize=True,
                      color=[0.85, 0.33, 0.10], label='Thrust direction')
h_start   = ax.plot([r_vals[0, 0]],  [r_vals[1, 0]],  [r_vals[2, 0]],
                    'go', markersize=8, markerfacecolor='g', label='Start')[0]
h_end     = ax.plot([r_vals[0, -1]], [r_vals[1, -1]], [r_vals[2, -1]],
                    'rs', markersize=8, markerfacecolor='r', label='End')[0]

ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
ax.set_title(f'Trajectory (m)  tf_opt = {tf_opt:.2f} s')
ax.legend(handles=[h_path, h_thrust, h_start, h_end], loc='best')

# ---- Figure 2: Kinematic states (2x2) ----
fig2, axes2 = plt.subplots(2, 2, num="Kinematic states")

ax = axes2[0, 0]; ax.grid(True)
for comp, lbl in zip(r_vals, ['r_x', 'r_y', 'r_z']):
    ax.plot(tv, comp, linewidth=1.2, label=lbl)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Position (m)'); ax.set_title('Position')
ax.legend(loc='best')

ax = axes2[0, 1]; ax.grid(True)
for comp, lbl in zip(v_vals, ['v_x', 'v_y', 'v_z']):
    ax.plot(tv, comp, linewidth=1.2, label=lbl)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Velocity (m/s)'); ax.set_title('Velocity')
ax.legend(loc='best')

ax = axes2[1, 0]; ax.grid(True)
for comp, lbl in zip(u_vals, ['u_x', 'u_y', 'u_z']):
    ax.plot(tv, comp, linewidth=1.2, label=lbl)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Acceleration (m/s²)'); ax.set_title('Commanded acceleration')
ax.legend(loc='best')

ax = axes2[1, 1]; ax.grid(True)
ax.plot(tv, m_vals, linewidth=1.2, label='m')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mass (kg)'); ax.set_title('Mass')
ax.legend(loc='best')

fig2.tight_layout()

# ---- Figure 3: Thrust profile (1x3) ----
fig3, axes3 = plt.subplots(1, 3, num="Thrust profile")

axes3[0].grid(True)
axes3[0].plot(tv, T_vals, linewidth=1.2, label='|T|')
axes3[0].set_xlabel('Time (s)'); axes3[0].set_ylabel('Thrust (N)')
axes3[0].set_title('Thrust magnitude'); axes3[0].legend(loc='best')

axes3[1].grid(True)
axes3[1].plot(tv, u_norms, linewidth=1.2, label='|u|')
axes3[1].set_xlabel('Time (s)'); axes3[1].set_ylabel('Acceleration (m/s²)')
axes3[1].set_title('Acceleration magnitude'); axes3[1].legend(loc='best')

axes3[2].grid(True)
axes3[2].plot(tv, u_dirs, linewidth=1.2, label=r'$\theta_u$')
axes3[2].set_xlabel('Time (s)'); axes3[2].set_ylabel('Angle (deg)')
axes3[2].set_title('Thrust direction (from vertical)'); axes3[2].legend(loc='best')

fig3.tight_layout()
plt.show()

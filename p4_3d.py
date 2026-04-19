# 3D convex powered-descent guidance in R^3.
# Python translation of p4_3d.m + plot_run3D.m.
# Requires: numpy, cvxpy, ecos, matplotlib
#
# Primary reference:
# [1] Acikmese & Ploen, "Convex programming approach to powered descent
#     guidance for mars landing," JGCD 30.5 (2007): 1353-1366.
# [2] Acikmese et al., "Enhancements on the convex programming based
#     powered descent guidance algorithm for mars landing," (2008).

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

# ---- Time grid ----
tf = 75.0
dt = 1.0
N  = int(tf / dt) + 1           # 76 nodes
tv = np.linspace(0.0, tf, N)    # [s]

# ---- Derived constants ----
# z === ln m,  u === T/m,  s === G/m  where |T| <= G
alpha = 1.0 / (Isp * g0 * np.cos(np.deg2rad(phi)))
r1    = min_throttle * T_max * np.cos(np.deg2rad(phi))  # lower thrust bound [N]
r2    = max_throttle * T_max * np.cos(np.deg2rad(phi))  # upper thrust bound [N]

# ---- CVXPY variables ----
r = cp.Variable((3, N))  # position [m]
v = cp.Variable((3, N))  # velocity [m/s]
u = cp.Variable((3, N))  # thrust acceleration [m/s^2]
z = cp.Variable(N)       # ln(mass) [nats]
s = cp.Variable(N)       # surrogate slack: |u[:,i]| <= s[i]

# ---- Objective: maximize ln of final mass, with thrust-rate regularization ----
# Small penalty on ||u[:,i+1] - u[:,i]||^2 breaks non-uniqueness on the optimal
# face and selects the smoothest thrust profile. lam should be small enough
# that final mass is essentially unchanged.
lam = 1e-4
du  = u[:, 1:] - u[:, :-1]            # shape (3, N-1)
objective = cp.Maximize(z[N - 1] - lam * cp.sum_squares(du))

constraints = []

# ---- Boundary conditions ----
constraints += [
    r[:, 0]     == r0,
    v[:, 0]     == v0,
    z[0]        == np.log(m_wet),
    r[:, N - 1] == rf,
    v[:, N - 1] == vf,
]

# ---- Dynamics (trapezoidal integration) ----
for i in range(N - 1):
    constraints += [
        v[:, i + 1] == v[:, i] + dt * g + (dt / 2) * (u[:, i] + u[:, i + 1]),
        r[:, i + 1] == (r[:, i]
                        + (dt / 2)    * (v[:, i] + v[:, i + 1])
                        + (dt**2 / 12) * (u[:, i + 1] - u[:, i])),
        z[i + 1]    == z[i] - (alpha * dt / 2) * (s[i] + s[i + 1]),
    ]

# ---- Thrust magnitude and mass flow limits ----
for i in range(N):
    # Feasible Taylor expansion reference point (scalar Python constants)
    z0_term = m_wet - alpha * r2 * i * dt
    z1_term = m_wet - alpha * r1 * i * dt
    z0      = np.log(z0_term)
    z1      = np.log(z1_term)
    mu_1    = r1 / z0_term
    mu_2    = r2 / z0_term

    dz = z[i] - z0  # affine CVXPY expression

    constraints += [
        cp.norm(u[:, i]) <= s[i],                            # SOC (|u| <= s)
        s[i] >= mu_1 * (1 - dz + 0.5 * cp.square(dz)),      # quadratic lower bound
        s[i] <= mu_2 * (1 - dz),                             # linear upper bound
        z[i] >= z0,                                          # mass floor
        z[i] <= z1,                                          # mass ceiling
    ]

# ---- Thrust pointing constraint (half-cone angle about vertical) ----
theta = 50  # deg
constraints += [
    u[2, :] >= s * np.cos(np.deg2rad(theta)),  # element-wise over all N nodes
]

# ---- No sub-surface flight ----
constraints += [r[2, :] >= -1]

# ---- Glide-slope cone (apex shifted h m below pad to avoid SOC collapse) ----
slope = 4   # deg
h     = 1   # m
for i in range(N):
    constraints += [
        cp.norm(r[0:2, i]) <= (r[2, i] + h) / np.tan(np.deg2rad(slope)),
    ]

# ---- Solve ----
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=True)

if problem.status not in ("optimal", "optimal_inaccurate"):
    raise RuntimeError(f"Solver failed with status: {problem.status}")

print(f"\nStatus     : {problem.status}")
print(f"Final z(N) : {z.value[N-1]:.6f}")
print(f"Fuel used  : {m_wet - np.exp(z.value[N-1]):.2f} kg")

# ---- Post-processing ----
r_vals  = r.value                    # (3, N)
v_vals  = v.value                    # (3, N)
u_vals  = u.value                    # (3, N)
z_vals  = z.value                    # (N,)

m_vals   = np.exp(z_vals)
u_norms  = np.linalg.norm(u_vals, axis=0)   # column norms, shape (N,)
u_dirs   = np.rad2deg(np.arctan2(
               np.sqrt(u_vals[0, :]**2 + u_vals[1, :]**2),
               u_vals[2, :]))                # thrust angle from vertical [deg]
T_vals   = u_norms * m_vals                 # thrust magnitude [N]

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
ax.set_title('Trajectory (m)')
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

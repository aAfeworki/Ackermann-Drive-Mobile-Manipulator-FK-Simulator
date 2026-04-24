import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Arm parameters
# ---------------------------
Lm1, Lm2, Lm3 = 0.1, 0.45, 0.3
t1, t2, t3 = 0.0, 0.0, 0.0

def get_arm_joints(t1, t2, t3):
    p0 = np.array([0, 0, 0])
    p1 = np.array([0, 0, Lm1])

    p2 = np.array([
        Lm2 * np.cos(t2) * np.cos(t1),
        Lm2 * np.cos(t2) * np.sin(t1),
        Lm1 + Lm2 * np.sin(t2)
    ])

    p3 = np.array([
        (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.cos(t1),
        (Lm2 * np.cos(t2) + Lm3 * np.cos(t2 + t3)) * np.sin(t1),
        Lm1 + Lm2 * np.sin(t2) + Lm3 * np.sin(t2 + t3)
    ])

    return p0, p1, p2, p3


# ---------------------------
# Ackermann parameters
# ---------------------------
L = 0.36
d = 0.60
Ts = 0.05

body_len = d + 0.20
body_wid = L + 0.20
body_h = 0.15

wheel_r = 0.07
wheel_w = 0.05

# state
X, Y, gamma = 0.0, 0.0, 0.0
V, w = 0.0, 0.0
alpha_l, alpha_r = 0.0, 0.0

path_x, path_y = [X], [Y]
trace_on = True
show_icr = True
_running = {"flag": True}

# ---------------------------
# Figure
# ---------------------------
plt.close("all")
fig = plt.figure(figsize=(11, 7))
ax = fig.add_axes([0.05, 0.08, 0.65, 0.88], projection='3d')


# ---------------------------
# Helpers
# ---------------------------
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def draw_box(cx, cy, cz):
    l, w_, h = body_len, body_wid, body_h
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2])
    y = np.array([w_ / 2, -w_ / 2, -w_ / 2, w_ / 2, w_ / 2])

    R = rot2d(gamma)
    pts = (R @ np.vstack((x, y))).T + np.array([cx, cy])

    ax.plot(pts[:, 0], pts[:, 1], cz, 'k')
    ax.plot(pts[:, 0], pts[:, 1], cz + h, 'k')

    for i in range(4):
        ax.plot([pts[i, 0], pts[i, 0]],
                [pts[i, 1], pts[i, 1]],
                [cz, cz + h], 'k')


def draw_wheel(cx, cy, heading):
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(-wheel_w / 2, wheel_w / 2, 10)
    theta, z = np.meshgrid(theta, z)

    x = wheel_r * np.cos(theta)
    y = z
    z_coords = wheel_r * np.sin(theta)

    R_steer = rot2d(heading)
    pts = np.vstack((x.flatten(), y.flatten()))
    rotated = R_steer @ pts

    Xw = rotated[0, :].reshape(x.shape)
    Yw = rotated[1, :].reshape(y.shape)

    base_pos = (rot2d(gamma) @ np.array([cx, cy])) + np.array([X, Y])

    ax.plot_surface(Xw + base_pos[0],
                    Yw + base_pos[1],
                    z_coords + wheel_r,
                    alpha=0.9)


def ackermann_steering(V, w):
    if abs(w) < 1e-9:
        return np.inf, 0.0, 0.0
    R_radius = V / w
    alpha_l = np.pi / 2 - np.arctan((R_radius + L / 2) / d)
    alpha_r = np.pi / 2 - np.arctan((R_radius - L / 2) / d)
    return R_radius, alpha_l, alpha_r


# ---------------------------
# Scene update
# ---------------------------
def update_scene():
    ax.cla()
    ax.set_xlim(X - 2, X + 2)
    ax.set_ylim(Y - 2, Y + 2)
    ax.set_zlim(0, 2)

    draw_box(X, Y, 0)

    offsets = {"fl": (d / 2, -L / 2), "fr": (d / 2, L / 2), "rl": (-d / 2, -L / 2), "rr": (-d / 2, L / 2)}
    for key, (ox, oy) in offsets.items():
        heading = gamma + (alpha_l if key == "fl" else alpha_r if key == "fr" else 0)
        draw_wheel(ox, oy, heading)

    if trace_on:
        ax.plot(path_x, path_y, 0, 'g--', alpha=0.5)

    # ICR visualization
    R_radius, _, _ = ackermann_steering(V, w)
    if show_icr and np.isfinite(R_radius):
        x_icr = X - R_radius * np.sin(gamma)
        y_icr = Y + R_radius * np.cos(gamma)
        ax.scatter(x_icr, y_icr, 0, c='r', marker='x', s=60)

    # Arm
    local_joints = get_arm_joints(t1, t2, t3)

    R_car = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    car_base = np.array([X, Y, body_h])
    gj = [(R_car @ p) + car_base for p in local_joints]

    for i in range(len(gj) - 1):
        ax.plot([gj[i][0], gj[i + 1][0]],
                [gj[i][1], gj[i + 1][1]],
                [gj[i][2], gj[i + 1][2]], 'r', lw=4, marker='o')

    # Info box 
    ee = gj[-1]
    info = (
        f"V: {V:.2f}\n"
        f"w: {w:.2f}\n"
        f"ICR: {R_radius:.2f}\n"
        f"αL: {alpha_l:.2f}, αR: {alpha_r:.2f}\n"
        f"X: {X:.2f}, Y: {Y:.2f}\n"
        f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}]"
    )

    ax.text2D(0.95, 0.95, info, transform=ax.transAxes,
              ha="right", va="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))


# ---------------------------
# Simulation 
# ---------------------------
def step():
    global X, Y, gamma, alpha_l, alpha_r
    if not _running["flag"]:
        return

    R_radius, alpha_l, alpha_r = ackermann_steering(V, w)
    gamma += w * Ts
    X += V * Ts * np.cos(gamma)
    Y += V * Ts * np.sin(gamma)

    if trace_on:
        path_x.append(X)
        path_y.append(Y)

    update_scene()
    fig.canvas.draw_idle()


# ---------------------------
# UI Controls
# ---------------------------
axcolor = "lightgoldenrodyellow"

s_V = Slider(fig.add_axes([0.75, 0.85, 0.18, 0.03], facecolor=axcolor), "V", -2, 2, valinit=0)
s_w = Slider(fig.add_axes([0.75, 0.80, 0.18, 0.03], facecolor=axcolor), "w", -2, 2, valinit=0)
s_t1 = Slider(fig.add_axes([0.75, 0.70, 0.18, 0.03], facecolor=axcolor), "θ1", -np.pi, np.pi, valinit=0)
s_t2 = Slider(fig.add_axes([0.75, 0.65, 0.18, 0.03], facecolor=axcolor), "θ2", -0.125, 3.265, valinit=0)
s_t3 = Slider(fig.add_axes([0.75, 0.60, 0.18, 0.03], facecolor=axcolor), "θ3", -np.pi / 2, np.pi / 2, valinit=0)

def update_vals(val):
    global V, w, t1, t2, t3
    V, w, t1, t2, t3 = s_V.val, s_w.val, s_t1.val, s_t2.val, s_t3.val

for s in [s_V, s_w, s_t1, s_t2, s_t3]:
    s.on_changed(update_vals)

# Play/Pause button
btn_run = Button(fig.add_axes([0.75, 0.50, 0.08, 0.05]), "Pause")

def toggle(event):
    _running["flag"] = not _running["flag"]
    btn_run.label.set_text("Pause" if _running["flag"] else "Play")

btn_run.on_clicked(toggle)

# Reset button
btn_reset = Button(fig.add_axes([0.85, 0.50, 0.08, 0.05]), "Reset")

def reset(event):
    global X, Y, gamma, path_x, path_y
    X, Y, gamma = 0, 0, 0
    path_x, path_y = [X], [Y]

btn_reset.on_clicked(reset)

# Checkboxes
ax_chk = fig.add_axes([0.75, 0.35, 0.18, 0.12])
chk = CheckButtons(ax_chk, ["Trace Path", "Show ICR"], [True, True])

def toggle_checks(label):
    global trace_on, show_icr
    if label == "Trace Path":
        trace_on = not trace_on
    elif label == "Show ICR":
        show_icr = not show_icr

chk.on_clicked(toggle_checks)

# Timer
timer = fig.canvas.new_timer(interval=int(Ts * 1000))
timer.add_callback(step)
timer.start()

update_scene()
plt.show()
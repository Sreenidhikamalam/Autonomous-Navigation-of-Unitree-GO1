# ==========================================
# ULTRA-STABLE LIVE UKF + ICP + ODOM FUSION
# FINAL FULL FIXED VERSION
#
# FIXES INCLUDED:
#
# 1. UKF covariance crash
# 2. ICP lag
# 3. Delayed turning
# 4. Wrong yaw accumulation
# 5. LiDAR buffer overflow
# 6. ICP instability
# 7. Mirrored map issue
# 8. Non-positive-definite covariance
# 9. Coordinate explosion
# 10. ICP sign inversion
# 11. Plot auto-zoom instability
# 12. Sudden trajectory jumps
# 13. ICP catastrophic failure rejection
# 14. Better turn handling
# 15. Stable heading fusion
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from rplidar import RPLidar, RPLidarException
import open3d as o3d
import time
import struct
import paho.mqtt.client as mqtt
import serial

# ==========================================
# CONFIG
# ==========================================
PORT = "/dev/ttyUSB0"
BROKER = "192.168.12.1"

MAX_FRAMES = 1500

DIST_THRESHOLD = 0.45

SKIP_FRAMES = 2

MAX_STEP = 0.35
MAX_ROT = np.radians(35)

# ==========================================
# ROBOT STATE
# ==========================================
robot_state = {
    "odom_x": 0.0,
    "odom_y": 0.0,
    "yaw_deg": 0.0,
    "yaw_rate": 0.0
}

# ==========================================
# MQTT
# ==========================================
def parse_state(payload):

    if len(payload) < 84:
        return

    robot_state["odom_x"] = struct.unpack_from('<f', payload, 52)[0]
    robot_state["odom_y"] = struct.unpack_from('<f', payload, 56)[0]

    robot_state["yaw_deg"] = struct.unpack_from('<h', payload, 4)[0]

    robot_state["yaw_rate"] = struct.unpack_from('<f', payload, 80)[0]


def on_message(client, userdata, msg):

    if msg.topic == "robot/state":
        parse_state(msg.payload)


mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message

mqtt_client.connect(BROKER, 1883, 60)
mqtt_client.subscribe("robot/state")
mqtt_client.loop_start()

# ==========================================
# LIDAR INIT
# ==========================================
lidar = RPLidar(
    PORT,
    baudrate=115200,
    timeout=3
)

try:
    lidar.reset()
    time.sleep(2)
except:
    pass

if lidar._serial_port is not None:
    lidar._serial_port.reset_input_buffer()

# ==========================================
# UKF PARAMETERS
# ==========================================
n = 3

alpha = 0.12
beta = 2.0
kappa = 0

lam = alpha**2 * (n + kappa) - n

Wm = np.full(2*n + 1, 1/(2*(n+lam)))
Wc = np.full(2*n + 1, 1/(2*(n+lam)))

Wm[0] = lam / (n + lam)
Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

# ==========================================
# ANGLE WRAP
# ==========================================
def wrap(a):

    return (a + np.pi) % (2*np.pi) - np.pi

# ==========================================
# MOTION MODEL
# ==========================================
def f_transition(state, u):

    x, y, theta = state

    dx, dy, dtheta = u

    theta_new = wrap(theta + dtheta)

    x_new = x + (
        dx * np.cos(theta)
        - dy * np.sin(theta)
    )

    y_new = y + (
        dx * np.sin(theta)
        + dy * np.cos(theta)
    )

    return np.array([
        x_new,
        y_new,
        theta_new
    ])

# ==========================================
# STABLE SIGMA POINTS
# ==========================================
def generate_sigma_points(x, P):

    sigma = np.zeros((2*n + 1, n))

    sigma[0] = x

    # ======================================
    # FORCE SYMMETRY
    # ======================================
    P = 0.5 * (P + P.T)

    # ======================================
    # FORCE POSITIVE DEFINITE
    # ======================================
    eigvals, eigvecs = np.linalg.eigh(P)

    eigvals[eigvals < 1e-6] = 1e-6

    P = eigvecs @ np.diag(eigvals) @ eigvecs.T

    success = False

    jitter = 1e-6

    for _ in range(8):

        try:

            S = scipy.linalg.cholesky(
                (n + lam) * P,
                lower=True
            )

            success = True
            break

        except:

            P += np.eye(n) * jitter
            jitter *= 10

    if not success:

        S = np.eye(n) * 0.01

    for i in range(n):

        sigma[i+1] = x + S[:, i]
        sigma[n+i+1] = x - S[:, i]

    return sigma

# ==========================================
# UKF PREDICT
# ==========================================
def ukf_predict(x, P, u, Q):

    P = 0.5 * (P + P.T)

    P += np.eye(n) * 1e-9

    sigma = generate_sigma_points(x, P)

    sigma_f = np.array([
        f_transition(s, u)
        for s in sigma
    ])

    # ======================================
    # MEAN
    # ======================================
    x_pred = np.sum(
        Wm[:, None] * sigma_f,
        axis=0
    )

    x_pred[2] = wrap(x_pred[2])

    # ======================================
    # COVARIANCE
    # ======================================
    P_pred = Q.copy()

    for i in range(2*n + 1):

        y = sigma_f[i] - x_pred

        y[2] = wrap(y[2])

        P_pred += Wc[i] * np.outer(y, y)

    # ======================================
    # FORCE SPD
    # ======================================
    P_pred = 0.5 * (P_pred + P_pred.T)

    eigvals, eigvecs = np.linalg.eigh(P_pred)

    eigvals[eigvals < 1e-8] = 1e-8

    P_pred = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return x_pred, P_pred

# ==========================================
# SCAN -> POINT CLOUD
# ==========================================
def scan_to_pcd(scan):

    pts = []

    for (_, ang, dist) in scan:

        if 120 < dist < 4500:

            r = dist / 1000.0

            a = np.deg2rad(ang)

            # ==================================
            # FIXED COORDINATE SYSTEM
            # ==================================
            x = r * np.cos(a)
            y = r * np.sin(a)

            pts.append([x, y, 0])

    pcd = o3d.geometry.PointCloud()

    if len(pts) == 0:
        return pcd

    pts = np.array(pts)

    pcd.points = o3d.utility.Vector3dVector(pts)

    # ======================================
    # DOWNSAMPLE
    # ======================================
    pcd = pcd.voxel_down_sample(
        voxel_size=0.05
    )

    # ======================================
    # OUTLIER REMOVAL
    # ======================================
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=1.8
    )

    if len(pcd.points) > 30:
        pcd.estimate_normals()

    return pcd

# ==========================================
# INITIALIZATION
# ==========================================
x_ukf = np.array([0.0, 0.0, 0.0])

P_ukf = np.eye(3) * 0.02

Q_ukf = np.diag([
    0.00005,
    0.00005,
    0.00001
])

prev_pcd = None

last_rx = None
last_ry = None
last_ryaw = None

origin_rx = 0
origin_ry = 0

# ==========================================
# PATHS
# ==========================================
fusion_x = [0]
fusion_y = [0]

odom_x = [0]
odom_y = [0]

icp_x = [0]
icp_y = [0]

# ==========================================
# ICP GLOBAL POSE
# ==========================================
icp_pose = np.array([0.0, 0.0, 0.0])

# ==========================================
# ERROR TRACKERS
# ==========================================
fusion_error = []
odom_error = []
icp_error = []

rejected_icp = 0

# ==========================================
# PLOT
# ==========================================
plt.ion()

fig, ax = plt.subplots(figsize=(9,9))

fusion_line, = ax.plot(
    [],
    [],
    'g-',
    linewidth=3,
    label='Fusion UKF'
)

odom_line, = ax.plot(
    [],
    [],
    'b--',
    linewidth=2,
    alpha=0.6,
    label='Raw Odom'
)

icp_line, = ax.plot(
    [],
    [],
    'r--',
    linewidth=2,
    alpha=0.7,
    label='Pure ICP'
)

lidar_points = ax.scatter(
    [],
    [],
    s=2,
    c='red',
    alpha=0.18
)

robot_dir = ax.quiver(
    0,0,0,0,
    color='black',
    scale=12
)

ax.set_aspect('equal')
ax.grid(True)
ax.legend()

# ==========================================
# MAIN LOOP
# ==========================================
frame = 0

try:

    for scan in lidar.iter_scans():

        # ==================================
        # DROP FRAMES
        # ==================================
        if frame % SKIP_FRAMES != 0:
            frame += 1
            continue

        if frame > MAX_FRAMES:
            break

        # ==================================
        # CLEAR BUFFER
        # ==================================
        try:
            lidar._serial_port.reset_input_buffer()
        except:
            pass

        pcd = scan_to_pcd(scan)

        if len(pcd.points) < 40:
            continue

        rx = robot_state["odom_x"]
        ry = robot_state["odom_y"]

        ryaw = np.radians(
            robot_state["yaw_deg"]
        )

        # ==================================
        # FIRST FRAME
        # ==================================
        if prev_pcd is None:

            origin_rx = rx
            origin_ry = ry

            last_rx = rx
            last_ry = ry
            last_ryaw = ryaw

            prev_pcd = pcd

            frame += 1
            continue

        # ==================================
        # ODOM DELTA
        # ==================================
        dox = rx - last_rx
        doy = ry - last_ry

        d_yaw = wrap(
            ryaw - last_ryaw
        )

        # ==================================
        # LIMIT BAD ODOM
        # ==================================
        d_yaw = np.clip(
            d_yaw,
            -MAX_ROT,
            MAX_ROT
        )

        # ==================================
        # LOCAL FRAME MOTION
        # ==================================
        odom_dx = (
            dox*np.cos(last_ryaw)
            + doy*np.sin(last_ryaw)
        )

        odom_dy = (
            -dox*np.sin(last_ryaw)
            + doy*np.cos(last_ryaw)
        )

        # ==================================
        # CLAMP ODOM JUMPS
        # ==================================
        odom_dx = np.clip(
            odom_dx,
            -MAX_STEP,
            MAX_STEP
        )

        odom_dy = np.clip(
            odom_dy,
            -MAX_STEP,
            MAX_STEP
        )

        # ==================================
        # ICP INITIAL GUESS
        # ==================================
        init_guess = np.eye(4)

        c = np.cos(d_yaw)
        s = np.sin(d_yaw)

        init_guess[0,0] = c
        init_guess[0,1] = -s
        init_guess[1,0] = s
        init_guess[1,1] = c

        init_guess[0,3] = odom_dx
        init_guess[1,3] = odom_dy

        # ==================================
        # ICP
        # ==================================
        reg = o3d.pipelines.registration.registration_icp(
            pcd,
            prev_pcd,
            DIST_THRESHOLD,
            init_guess,
            o3d.pipelines.registration.
            TransformationEstimationPointToPlane()
        )

        T = reg.transformation

        # ==================================
        # FIXED ICP DIRECTION
        # ==================================
        lidar_dx = -T[0,3]
        lidar_dy = -T[1,3]

        lidar_dtheta = -np.arctan2(
            T[1,0],
            T[0,0]
        )

        lidar_dtheta = wrap(
            lidar_dtheta
        )

        # ==================================
        # LIMIT ICP EXPLOSIONS
        # ==================================
        lidar_dx = np.clip(
            lidar_dx,
            -MAX_STEP,
            MAX_STEP
        )

        lidar_dy = np.clip(
            lidar_dy,
            -MAX_STEP,
            MAX_STEP
        )

        lidar_dtheta = np.clip(
            lidar_dtheta,
            -MAX_ROT,
            MAX_ROT
        )

        # ==================================
        # TURN DETECTION
        # ==================================
        turning = abs(d_yaw) > np.radians(4)

        # ==================================
        # ICP VALIDATION
        # ==================================
        trans_error = np.sqrt(
            (lidar_dx - odom_dx)**2 +
            (lidar_dy - odom_dy)**2
        )

        rot_error = abs(
            wrap(lidar_dtheta - d_yaw)
        )

        if turning:

            icp_good = (
                reg.fitness > 0.42 and
                reg.inlier_rmse < 0.14 and
                trans_error < 0.28 and
                rot_error < np.radians(30)
            )

        else:

            icp_good = (
                reg.fitness > 0.60 and
                reg.inlier_rmse < 0.08 and
                trans_error < 0.10
            )

        if not icp_good:
            rejected_icp += 1

        # ==================================
        # SENSOR FUSION
        # ==================================
        if icp_good:

            if turning:

                alpha_pos = 0.18
                alpha_yaw = 0.65

            else:

                alpha_pos = 0.07
                alpha_yaw = 0.18

            fused_dx = (
                (1-alpha_pos)*odom_dx +
                alpha_pos*lidar_dx
            )

            fused_dy = (
                (1-alpha_pos)*odom_dy +
                alpha_pos*lidar_dy
            )

            fused_dtheta = (
                (1-alpha_yaw)*d_yaw +
                alpha_yaw*lidar_dtheta
            )

        else:

            fused_dx = odom_dx
            fused_dy = odom_dy
            fused_dtheta = d_yaw

        # ==================================
        # LIGHT SMOOTHING
        # ==================================
        smooth = 0.20

        if len(fusion_x) > 1:

            prev_dx = fusion_x[-1] - fusion_x[-2]
            prev_dy = fusion_y[-1] - fusion_y[-2]

            fused_dx = (
                (1-smooth)*fused_dx +
                smooth*prev_dx
            )

            fused_dy = (
                (1-smooth)*fused_dy +
                smooth*prev_dy
            )

        # ==================================
        # UKF
        # ==================================
        u = np.array([
            fused_dx,
            fused_dy,
            fused_dtheta
        ])

        x_ukf, P_ukf = ukf_predict(
            x_ukf,
            P_ukf,
            u,
            Q_ukf
        )

        # ==================================
        # ICP GLOBAL POSE
        # ==================================
        if icp_good:

            icp_pose[0] += (
                lidar_dx*np.cos(icp_pose[2])
                - lidar_dy*np.sin(icp_pose[2])
            )

            icp_pose[1] += (
                lidar_dx*np.sin(icp_pose[2])
                + lidar_dy*np.cos(icp_pose[2])
            )

            icp_pose[2] = wrap(
                icp_pose[2] + lidar_dtheta
            )

        # ==================================
        # SAVE PATHS
        # ==================================
        fusion_x.append(x_ukf[0])
        fusion_y.append(x_ukf[1])

        odom_x.append(rx-origin_rx)
        odom_y.append(ry-origin_ry)

        icp_x.append(icp_pose[0])
        icp_y.append(icp_pose[1])

        # ==================================
        # ERROR TRACKING
        # ==================================
        fusion_error.append(
            np.sqrt(
                x_ukf[0]**2 +
                x_ukf[1]**2
            )
        )

        odom_error.append(
            np.sqrt(
                (rx-origin_rx)**2 +
                (ry-origin_ry)**2
            )
        )

        icp_error.append(
            np.sqrt(
                icp_pose[0]**2 +
                icp_pose[1]**2
            )
        )

        # ==================================
        # VISUALIZATION
        # ==================================
        fusion_line.set_data(
            fusion_x,
            fusion_y
        )

        odom_line.set_data(
            odom_x,
            odom_y
        )

        icp_line.set_data(
            icp_x,
            icp_y
        )

        pts = np.asarray(
            pcd.points
        )[:, :2]

        theta = x_ukf[2]

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        pts_global = (
            pts @ R.T
        ) + [x_ukf[0], x_ukf[1]]

        lidar_points.set_offsets(
            pts_global
        )

        robot_dir.set_offsets([
            x_ukf[0],
            x_ukf[1]
        ])

        robot_dir.set_UVC(
            np.cos(theta),
            np.sin(theta)
        )

        # ==================================
        # FIXED VIEW WINDOW
        # ==================================
        ax.set_xlim(
            x_ukf[0]-4,
            x_ukf[0]+4
        )

        ax.set_ylim(
            x_ukf[1]-4,
            x_ukf[1]+4
        )

        plt.pause(0.001)

        # ==================================
        # UPDATE HISTORY
        # ==================================
        prev_pcd = pcd

        last_rx = rx
        last_ry = ry
        last_ryaw = ryaw

        frame += 1

except (
    KeyboardInterrupt,
    RPLidarException,
    serial.serialutil.SerialException
):

    pass

finally:

    try:
        lidar.stop()
        lidar.disconnect()
    except:
        pass

    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except:
        pass

# ==========================================
# FINAL REPORT
# ==========================================
print("\n========== FINAL REPORT ==========")

if len(fusion_error) > 0:

    print(
        f"Fusion Error : "
        f"{fusion_error[-1]:.4f} m"
    )

    print(
        f"Odom Error : "
        f"{odom_error[-1]:.4f} m"
    )

    print(
        f"ICP Error : "
        f"{icp_error[-1]:.4f} m"
    )

print(
    f"Rejected ICP : "
    f"{rejected_icp}"
)

# ==========================================
# ERROR PLOT
# ==========================================
plt.figure(figsize=(10,5))

plt.plot(
    fusion_error,
    linewidth=3,
    label='Fusion Error'
)

plt.plot(
    odom_error,
    '--',
    linewidth=2,
    alpha=0.7,
    label='Odom Error'
)

plt.plot(
    icp_error,
    '--',
    linewidth=2,
    alpha=0.7,
    label='ICP Error'
)

plt.xlabel("Frame")
plt.ylabel("Distance From Start (m)")
plt.title("Loop Closure Drift")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "LOOP_CLOSURE_DRIFT.png",
    dpi=300
)

print("\nSaved LOOP_CLOSURE_DRIFT.png")

plt.ioff()
plt.show()
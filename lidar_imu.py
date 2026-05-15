import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar, RPLidarException
import open3d as o3d
import time
import struct
import paho.mqtt.client as mqtt

# ===== CONFIGURATION =====
PORT = "/dev/ttyUSB0"  # Change to "COM31" if on Windows
MAX_FRAMES = 1500
DIST_THRESHOLD = 0.5  
BROKER = "192.168.12.1" # Robot WiFi IP

# ===== ROBOT SENSOR DATA (MQTT) =====
robot_state = {"odom_x": 0.0, "odom_y": 0.0, "yaw_deg": 0.0, "yaw_rate": 0.0}

def parse_state(payload):
    if len(payload) < 84: return
    robot_state["odom_x"] = struct.unpack_from('<f', payload, 52)[0]
    robot_state["odom_y"] = struct.unpack_from('<f', payload, 56)[0]
    robot_state["yaw_deg"] = struct.unpack_from('<h', payload, 4)[0]
    robot_state["yaw_rate"] = struct.unpack_from('<f', payload, 80)[0]

def on_message(client, userdata, msg):
    if msg.topic == "robot/state":
        parse_state(msg.payload)

# Start MQTT Background Thread
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER, 1883, 60)
mqtt_client.subscribe("robot/state")
mqtt_client.loop_start()

# ===== INIT LIDAR =====
lidar = RPLidar(PORT, baudrate=115200, timeout=3)
try:
    lidar.reset()
    time.sleep(2)
except:
    pass
if lidar._serial_port is not None:
    lidar._serial_port.reset_input_buffer()

# ===== DATA STORAGE =====
fgo_registry = []
current_pose = np.eye(4)
prev_pcd = None

# ===== PREPARE PLOT =====
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
ray_line, = ax.plot([], [], color='lightblue', alpha=0.3, linewidth=0.5, zorder=1)
wall_dots = ax.scatter([], [], s=2, c='black', zorder=2)
traj_line, = ax.plot([], [], 'r-', linewidth=1.5, label="Lidar Trajectory", zorder=3)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.legend()
ax.set_title("Live Data: Lidar ICP + Robot IMU")

def scan_to_pcd(scan):
    pts = []
    for (_, ang, dist) in scan:
        if dist > 0:
            r = dist / 1000.0
            a = np.deg2rad(ang)
            pts.append([r * np.cos(a), r * np.sin(a), 0])
    pcd = o3d.geometry.PointCloud()
    if len(pts) > 0:
        pcd.points = o3d.utility.Vector3dVector(np.array(pts))
    return pcd

print("Connecting to Lidar and Robot IMU...")

frame_count = 0
try:
    for scan in lidar.iter_scans():
        if frame_count > MAX_FRAMES:
            break
            
        pcd = scan_to_pcd(scan)
        if len(pcd.points) < 10:
            continue

        t_guess = np.eye(4)
        dx, dy, dtheta = 0.0, 0.0, 0.0

        if prev_pcd is not None:
            reg = o3d.pipelines.registration.registration_icp(
                pcd, prev_pcd, DIST_THRESHOLD, t_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            rel_t = reg.transformation
            dx = rel_t[0, 3]
            dy = rel_t[1, 3]
            dtheta = np.arctan2(rel_t[1, 0], rel_t[0, 0])
            current_pose = current_pose @ rel_t

        gx = current_pose[0, 3]
        gy = current_pose[1, 3]
        gtheta = np.arctan2(current_pose[1, 0], current_pose[0, 0])

        # Grab latest IMU/Odom data from MQTT background thread
        rx = robot_state["odom_x"]
        ry = robot_state["odom_y"]
        ryaw = robot_state["yaw_deg"]
        rrate = robot_state["yaw_rate"]

        # LIVE TERMINAL PRINT
        print(f"Lidar: [X:{gx:+.2f} Y:{gy:+.2f}] | Robot IMU: [Yaw:{ryaw:+.1f}deg Rate:{rrate:+.2f}]")

        # Save everything to one row!
        fgo_registry.append([time.time(), gx, gy, gtheta, dx, dy, dtheta, rx, ry, ryaw, rrate])

        # Visualization updates
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            wall_dots.set_offsets(pts[:, :2])
            rays_x = np.zeros(len(pts) * 3)
            rays_y = np.zeros(len(pts) * 3)
            rays_x[1::3] = pts[:, 0]
            rays_y[1::3] = pts[:, 1]
            rays_x[2::3] = np.nan
            rays_y[2::3] = np.nan
            ray_line.set_data(rays_x, rays_y)

        traj_arr = np.array(fgo_registry)
        traj_line.set_data(traj_arr[:, 1], traj_arr[:, 2])

        ax.set_xlim(gx - 5, gx + 5)
        ax.set_ylim(gy - 5, gy + 5)
        
        plt.pause(0.001)
        prev_pcd = pcd
        frame_count += 1

except (RPLidarException, KeyboardInterrupt) as e:
    print(f"\nStopping: {e}")

finally:
    print("\n--- STOPPING RECORDING ---")
    
    # SAVE THE MASTER CSV
    if fgo_registry:
        data_to_save = np.array(fgo_registry)
        header = "timestamp,lidar_x,lidar_y,lidar_theta,lidar_dx,lidar_dy,lidar_dtheta,robot_odom_x,robot_odom_y,robot_yaw_deg,robot_yaw_rate"
        np.savetxt("lidar_and_imu_data3.csv", data_to_save, delimiter=",", header=header)
        print(f"SUCCESS! Saved {len(data_to_save)} frames to 'lidar_and_imu_data3.csv'")
    else:
        print("ERROR: No data collected.")

    try:
        lidar.stop()
        lidar.disconnect()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except:
        pass

plt.ioff()
plt.show()
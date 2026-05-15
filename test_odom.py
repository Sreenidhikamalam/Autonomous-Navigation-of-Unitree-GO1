"""
test_odom.py - Execute a sequence of walk/turn commands using Go1 odometry
Usage:
  python3 test_odom.py forward 2 right 90 forward 3 left 90 backward 1

Commands:
  forward/backward/left/right <meters>  - walk in that direction
  turnleft/turnright <degrees>          - turn in place

Example - walk a 2m square:
  python3 test_odom.py forward 2 turnright 90 forward 2 turnright 90 forward 2 turnright 90 forward 2
"""

import sys, time, math, struct
import paho.mqtt.client as mqtt

BROKER = "192.168.12.1"

# Walk parameters
WALK_SPEED = 0.3
STOP_EARLY = 0.15
HEADING_KP = 0.002
LATERAL_KP = 0.15

# Turn parameters
YAW_SPEED = 0.2

# Odometry state
odom = {"x": None, "y": None, "yaw_deg": 0, "yaw_rate": 0.0}
t_last_walk = [0.0]


def pack(f):
    return list(struct.pack('f', float(f)))


def parse_state(payload):
    if len(payload) < 84:
        return
    odom["x"] = struct.unpack_from('<f', payload, 52)[0]
    odom["y"] = struct.unpack_from('<f', payload, 56)[0]
    odom["yaw_deg"] = struct.unpack_from('<h', payload, 4)[0]
    odom["yaw_rate"] = struct.unpack_from('<f', payload, 80)[0]


def on_message(client, userdata, msg):
    if msg.topic == "robot/state":
        parse_state(msg.payload)


def stop_robot(client):
    client.publish("controller/stick", bytearray([0] * 16)).wait_for_publish()


def send_cmd(client, forward=0.0, lateral=0.0, yaw=0.0):
    payload = pack(lateral) + pack(yaw) + pack(0.0) + pack(forward)
    client.publish("controller/stick", bytearray(payload), qos=0).wait_for_publish()


def keep_walk(client):
    """Republish walk gait every 1 second."""
    now = time.time()
    if now - t_last_walk[0] > 1.0:
        client.publish("controller/action", "walk")
        t_last_walk[0] = now


def do_walk(client, direction, distance):
    """Walk a given distance in a direction with heading + lateral correction."""
    if direction == "forward":
        fwd, lat = WALK_SPEED, 0.0
    elif direction == "backward":
        fwd, lat = -WALK_SPEED, 0.0
    elif direction == "left":
        fwd, lat = 0.0, -WALK_SPEED
    elif direction == "right":
        fwd, lat = 0.0, WALK_SPEED
    else:
        return

    start_x = odom["x"]
    start_y = odom["y"]
    start_yaw = odom["yaw_deg"]
    effective = distance - STOP_EARLY
    yaw_rad = math.radians(start_yaw)

    print(f"  [{direction} {distance}m] start=({start_x:.2f},{start_y:.2f}) yaw={start_yaw}deg")

    while True:
        keep_walk(client)

        dx = odom["x"] - start_x
        dy = odom["y"] - start_y
        dist = math.hypot(dx, dy)

        # Cross-track
        if direction in ("forward", "backward"):
            ct = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        else:
            ct = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)

        print(f"\r    dist={dist:.3f}/{distance}m  cross={ct:+.3f}m  yaw={odom['yaw_deg']}deg  ",
              end="", flush=True)

        if dist >= effective:
            print(f"\n  Done! dist={dist:.3f}m")
            stop_robot(client)
            time.sleep(0.5)
            return

        # Heading correction
        yaw_err = odom["yaw_deg"] - start_yaw
        while yaw_err > 180: yaw_err -= 360
        while yaw_err < -180: yaw_err += 360
        yaw_corr = max(-0.3, min(0.3, HEADING_KP * yaw_err))

        # Lateral correction
        if direction in ("forward", "backward"):
            cross_track = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
            lat_corr = max(-0.2, min(0.2, -LATERAL_KP * cross_track))
        elif direction in ("left", "right"):
            cross_track = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            lat_corr = max(-0.2, min(0.2, -LATERAL_KP * cross_track))
        else:
            lat_corr = 0.0

        send_cmd(client, forward=fwd, lateral=lat + lat_corr, yaw=yaw_corr)
        time.sleep(0.05)


def do_turn(client, direction, degrees):
    """Turn in place using yaw_rate integration."""
    start_heading = odom["yaw_deg"]
    yaw_cmd = -YAW_SPEED if direction == "turnleft" else YAW_SPEED

    print(f"  [{direction} {degrees}deg] start_heading={start_heading}deg")

    t_last = time.time()
    cumulative = 0.0

    while True:
        keep_walk(client)

        now = time.time()
        dt = now - t_last
        t_last = now
        cumulative += math.degrees(odom["yaw_rate"]) * dt
        turned = abs(cumulative)

        print(f"\r    turned={turned:.1f}/{degrees}deg  heading={odom['yaw_deg']}deg  rate={odom['yaw_rate']:+.3f}  ",
              end="", flush=True)

        if turned >= degrees:
            print(f"\n  Done! turned={turned:.1f}deg  heading={odom['yaw_deg']}deg")
            stop_robot(client)
            time.sleep(0.5)
            return

        send_cmd(client, yaw=yaw_cmd)
        time.sleep(0.05)


def parse_commands(args):
    """Parse command line into list of (action, value) tuples."""
    commands = []
    i = 0
    while i < len(args):
        action = args[i].lower()
        if action in ("forward", "backward", "left", "right", "turnleft", "turnright"):
            if i + 1 < len(args):
                value = float(args[i + 1])
                commands.append((action, value))
                i += 2
            else:
                print(f"Missing value for {action}")
                i += 1
        else:
            print(f"Unknown command: {action}")
            i += 1
    return commands


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 test_odom.py forward 2 turnright 90 forward 3")
        print("Commands: forward/backward/left/right <meters>")
        print("          turnleft/turnright <degrees>")
        return

    commands = parse_commands(sys.argv[1:])
    if not commands:
        print("No valid commands")
        return

    print(f"=== Odom Command Sequence ({len(commands)} commands) ===")
    for i, (action, val) in enumerate(commands):
        unit = "deg" if "turn" in action else "m"
        print(f"  {i+1}. {action} {val}{unit}")

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER, 1883, 60)
    client.subscribe("robot/state")
    client.loop_start()

    print("\nWaiting for odometry...")
    for _ in range(50):
        if odom["x"] is not None:
            break
        time.sleep(0.1)

    if odom["x"] is None:
        print("ERROR: No odometry data received")
        client.loop_stop()
        return

    origin_x = odom["x"]
    origin_y = odom["y"]
    print(f"Origin: ({origin_x:.2f}, {origin_y:.2f}) yaw={odom['yaw_deg']}deg")

    client.publish("controller/action", "walk").wait_for_publish()
    print("Waiting 4s for gait init...\n")
    time.sleep(4)
    t_last_walk[0] = time.time()

    try:
        for i, (action, val) in enumerate(commands):
            print(f"--- Command {i+1}/{len(commands)} ---")
            if action in ("forward", "backward", "left", "right"):
                do_walk(client, action, val)
            elif action in ("turnleft", "turnright"):
                do_turn(client, action, val)

        # Final report
        dx = odom["x"] - origin_x
        dy = odom["y"] - origin_y
        print(f"\n=== Complete ===")
        print(f"Total displacement: dx={dx:+.3f} dy={dy:+.3f} dist={math.hypot(dx, dy):.3f}m")
        print(f"Final heading: {odom['yaw_deg']}deg")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        stop_robot(client)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()

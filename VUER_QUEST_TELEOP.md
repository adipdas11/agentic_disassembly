# Meta Quest teleoperation with Vuer

This workspace can run Quest hand tracking directly against Ubuntu. Windows,
Unity, and ROS-TCP-Endpoint are not part of this path.

## Architecture

```text
Quest Browser (WebXR)
  ├─ native low-latency tracked hand meshes
  ├─ 25 joint transforms per hand ── WSS ──> vuer_quest_bridge
  │                                      ├─ wrist PoseStamped
  │                                      ├─ landmarks PoseArray
  │                                      ├─ pinch/fist/pinky-pinch
  │                                      └─ normalized gripper aperture
  │                                                   │
  │                                                   v
  │                                      exotica_arm_teleop -> robot
  │
  └─ two head-locked video panels <── WebRTC/H.264 <── compressed ROS images
                                                        ^
                                                        │
                                             two RealSense cameras
```

Vuer supplies a Python server and a browser WebXR client. Its `Hands` component
streams 25 WebXR joint matrices for each hand. The bridge converts the WebXR
axes (+X right, +Y up, -Z forward) to the teleop ROS axes (+X forward, +Y
left, +Z up), then publishes the interfaces already consumed by this project.

The camera panels are independent workcell views. They intentionally are not
sent one per eye: the cameras are not a calibrated binocular pair, so treating
them as stereo would create incorrect depth and an uncomfortable display.

## ROS interfaces

The bridge publishes, for both `left` and `right`:

| Topic suffix | Type | Meaning |
| --- | --- | --- |
| `wrist` | `geometry_msgs/PoseStamped` | Wrist position with canonical anatomical palm orientation |
| `landmarks` | `geometry_msgs/PoseArray` | All 25 joints in WebXR order |
| `pinch` | `std_msgs/Bool` | Thumb-index pinch |
| `pinky_pinch` | `std_msgs/Bool` | Thumb-pinky pinch with hysteresis |
| `fist` | `std_msgs/Bool` | Vuer squeeze state |
| `gripper_aperture` | `std_msgs/Float32` | Hand-size-normalized value, 0–1 |

Every topic is below `/teleop_hand_tracking/<hand>/`. The `PoseArray` ordering
is the WebXR `XRHand` ordering documented by Vuer, from wrist at index 0 through
pinky tip at index 24.

Vuer orientation uses the WebXR wrist joint frame directly. WebXR defines its
longitudinal and palm-normal axes, so this avoids amplifying independent
position noise from several short palm bones. The fixed basis remap produces
the shared right-handed anatomical axes: X toward the fingers, Y across the
palm, and Z out through the palm. MediaPipe reconstructs that same frame from
its metric wrist and MCP landmarks because it does not provide joint poses.
UF850 pitch and yaw tracking are enabled by default; forearm twist remains
opt-in. When only selected axes are enabled, their hand-local rotation-vector
components are retained without Euler-angle coupling near 90-degree motions.
The default hand-to-tool alignment assumes a bottle grasp: wrist-to-fingers
follows TCP approach and the palm normal (thumb-index opposition) follows the
RG6 jaw-separation axis. Consequently, wrist flexion drives gripper up/down
tilt and side-to-side wrist deviation drives left/right tilt.

Quest wrist translation and anatomical rotation use one-to-one endpoint gains.
A 90-degree wrist rotation therefore commands 90 degrees at the gripper. A
continuous 10 mm per-axis translation dead zone rejects tracking noise; motion
beyond it remains continuous because the dead-zone width is subtracted before
scaling. The bridge already converts WebXR forward (`-Z`) to robot-base forward
(`+X`), so the controller must not invert that axis a second time. Filtering
changes only arrival speed, not the final pose.

The source pose is located at the human wrist, but EXOTica solves at each
gripper's fingertip TCP. The controller converts through a model-derived pivot
offset (RG6: 280 mm; xArm gripper: 170 mm), keeping the robot gripper mount
stationary during a pure wrist rotation. Without this conversion the mount
would visibly sweep around the fingertip TCP even though the commanded TCP
position itself remained fixed.

The Quest display shows Vuer's native tracked hand meshes by default. All 25
joints are streamed directly to ROS without sending a second visualization
back to the headset, avoiding an unnecessary Quest-to-Ubuntu-to-Quest round
trip. Hide the native meshes while retaining ROS tracking when desired:

```bash
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  hide_hand_meshes:=true
```

If the browser disconnects or a hand disappears, wrist messages stop. The
existing `tracking_timeout_sec` stops target updates after one second. Both arms
also start disabled and must be enabled explicitly from the control panel or a
ROS service after tracking is stable.

## One-time setup

Quest and Ubuntu must be on the same LAN. A dedicated 5 GHz or 6 GHz access
point is preferable for consistent camera latency. Give Ubuntu a DHCP
reservation/static address because the TLS certificate and Quest bookmark use
that address.

From this repository, create a local certificate containing the Ubuntu LAN IP:

```bash
./setup_vuer_tls.sh 192.168.1.100
```

Replace the example address. The script stores the key outside the repository,
at `<ros-workspace>/.vuer/tls/`; it refuses to overwrite an existing key. If the
Ubuntu address changes, remove that directory deliberately and regenerate it.

Build the portable image once:

```bash
./setup_teleop_docker.sh
```

The image pins `vuer[webrtc]==0.1.6`; this supplies the WebXR server, `aiortc`,
and PyAV/H.264 camera transport for ROS Humble's Python 3.10 environment. The
Compose service uses host networking, which is important for same-LAN WebRTC
ICE candidates as well as robot access.

## Run

Start the prepared container:

```bash
./run_teleop_docker.sh
```

First validate with fake hardware. Inside the container:

```bash
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  hardware_type:=fake \
  use_rviz:=true \
  enable_cameras:=true
```

On Quest, open the address printed by the bridge:

```text
https://192.168.1.100:8012/?ws=wss://192.168.1.100:8012
```

Accept the local certificate warning, reload the full URL if necessary, select
the browser's **Enter VR** control, and grant hand-tracking permission. This is
self-hosted by the Ubuntu process, so it does not require the `vuer.ai` hosted
client or an internet tunnel after the page is loaded.

Check the data before enabling either arm:

```bash
ros2 topic hz /teleop_hand_tracking/right/wrist
ros2 topic echo /teleop_hand_tracking/right/landmarks --once
ros2 topic hz /camera1/realsense_camera/color/image_raw/compressed
ros2 topic hz /camera2/realsense_camera/color/image_raw/compressed
```

For the real robot, keep both hands visible and stationary, then use:

```bash
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  hardware_type:=real \
  use_rviz:=true \
  enable_uf850:=true \
  enable_xarm5:=true \
  uf850_hand:=right \
  xarm5_hand:=left \
  enable_cameras:=true
```

Enable from the Tk control panel only when the initial hand and robot pose are
safe. The equivalent command is:

```bash
ros2 service call /exotica_arm_teleop/set_all_arms_enabled \
  std_srvs/srv/SetBool "{data: true}"
```

For the UF850-only laptop-disassembly workflow, including all updated wrist
axes but no RealSense camera nodes or Quest video panels, use:

```bash
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  hardware_type:=real \
  use_rviz:=true \
  enable_uf850:=true \
  enable_xarm5:=false \
  uf850_hand:=right \
  xarm5_hand:=left \
  uf850_track_orientation:=false \
  uf850_track_yaw:=true \
  uf850_track_pitch:=true \
  uf850_track_roll:=true \
  enable_cameras:=false \
  enable_camera_streams:=false
```

Disable immediately with the same call using `false`.

## Camera selection and tuning

List the attached RealSense serial numbers:

```bash
rs-enumerate-devices | grep -E 'Name|Serial Number'
```

Override the stored camera serials at launch when moving the stack to different
hardware:

```bash
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  hardware_type:=fake \
  camera1_serial:=_SERIAL_ONE \
  camera2_serial:=_SERIAL_TWO
```

The leading underscore is required by the RealSense ROS parameter convention
used in this launch file. Camera topics, frame rate, bitrate, resolution, and
panel placement are in `arm_teleop/config/vuer_quest_bridge.yaml`.

Useful launch combinations:

```bash
# Cameras already run elsewhere; subscribe and stream without launching them.
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  enable_cameras:=false enable_camera_streams:=true

# Hand tracking only, with no video encoding or camera subscriptions.
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  enable_cameras:=false enable_camera_streams:=false

# Only one physical camera node; the unused second panel remains black unless
# enable_camera_streams is disabled or its topic is changed in the YAML.
ros2 launch arm_teleop vuer_quest_teleop.launch.py \
  enable_camera1:=true enable_camera2:=false
```

For one-camera operation, make a small local copy of the Vuer YAML and set the
second topic to another desired image source, or disable camera streaming. A
future UI can add panel toggles without changing the ROS hand interface.

## Native Ubuntu alternative

Docker is the recommended reproducible route. A native installation must use
the ROS Humble Python interpreter (3.10), not this repository's LeRobot Python
3.12 environment, because binary `rclpy` modules are Python-version-specific:

```bash
python3.10 -m venv --system-site-packages .venv-ros-vuer
source .venv-ros-vuer/bin/activate
python -m pip install 'vuer[webrtc]==0.1.6' opencv-python 'numpy<2'
source /opt/ros/humble/setup.bash
source /home/adip/workspace/disassembly_ws/install/setup.bash
export VUER_CERT_FILE=/home/adip/workspace/disassembly_ws/.vuer/tls/cert.pem
export VUER_KEY_FILE=/home/adip/workspace/disassembly_ws/.vuer/tls/key.pem
ros2 launch arm_teleop vuer_quest_teleop.launch.py hardware_type:=fake
```

## Troubleshooting

- No **Enter VR** button usually means the page or websocket is not secure.
  Confirm both `https://` and `wss://`, and accept the certificate at the exact
  Ubuntu IP used in the URL.
- A TLS hostname/IP error means the Ubuntu address differs from the certificate
  SAN. Regenerate the certificate for the new static address.
- Hand topics require entering immersive VR and granting hand permission; merely
  opening the page does not start `HAND_MOVE` events.
- Black video panels with working hand topics usually mean the compressed ROS
  topics are absent. Check `ros2 topic list`, the serial numbers, and USB 3
  connectivity.
- If hand topics work but WebRTC does not, test
  `https://<ubuntu-ip>:8012/webrtc/debug?url=https://<ubuntu-ip>:8012/webrtc/offer/camera1`
  and check host firewall rules. Host networking is already enabled in Compose.
- Reduce `camera_bitrate_bps`, `camera_fps`, or resolution in the Vuer YAML if
  Wi-Fi latency or dropped frames are high.
- Do not expose port 8012 to an untrusted network. Vuer does not provide robot
  authorization here; use a dedicated LAN, restrict the firewall to the Quest,
  and retain the physical robot emergency stop.

## Upstream references

- [Vuer concepts and Quest WebXR support](https://docs.vuer.ai/en/latest/tutorials/basics.html)
- [Vuer hand tracking example](https://docs.vuer.ai/en/latest/examples/vr_xr/hand_tracking.html)
- [Vuer TLS requirements and LAN options](https://docs.vuer.ai/en/latest/tutorials/basics/ssl_proxy_webxr.html)
- [Vuer source and integrated WebRTC implementation](https://github.com/vuer-ai/vuer)
- [OpenTeleVision reference implementation](https://github.com/OpenTeleVision/TeleVision)

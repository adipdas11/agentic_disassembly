#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, String
import serial
import glob
import json
import os
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


COMMAND_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)
STATUS_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

class ToolCommander(Node):
    def __init__(self):
        super().__init__('tool_commander')

        # --- Parameters ---
        self.declare_parameter('port', 'auto')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('allow_fake', False)
        
        self.configured_port = str(self.get_parameter('port').value)
        self.port = self.configured_port
        self.baud = self.get_parameter('baud').value
        self.allow_fake = bool(self.get_parameter('allow_fake').value)

        # --- Hardware Connection ---
        self.ser = None
        self.fake = False
        self.last_error = ""
        
        # Use absolute names because this node is launched from a nested MoveIt
        # process while the disassembly skill is launched by the parent process.
        # Relative names silently diverge if either launch later adds a namespace.
        self.status_pub = self.create_publisher(String, '/tool_status', STATUS_QOS)

        # --- Subscribers ---
        # 1 = Screw, 0 = Stop, -1 = Unscrew
        # 2 = Grab (155°), 3 = Release (0°)
        self.create_subscription(Int8, '/tool_cmd', self.handle_cmd, COMMAND_QOS)
        self.create_timer(1.0, self._connection_tick)

        self.current_cmd = 0
        self.command_seq = 0
        self.request_seq = 0
        self.last_requested_cmd = None
        self.last_command_ok = None
        self._try_connect()
        self._publish_status()

    def _candidate_ports(self):
        configured = self.configured_port.strip()
        if configured and configured.lower() != "auto":
            return [configured]

        ports = []
        # The screwdriver is a MicroPython CDC/ACM device.  Do not include
        # arbitrary ttyUSB devices here: on this system ttyUSB0 is the FT300
        # RS-485 adapter and opening it at 115200 would steal it from the force
        # sensor driver.
        patterns = [
            "/dev/serial/by-id/*MicroPython*",
            "/dev/ttyACM*",
        ]
        for pattern in patterns:
            ports.extend(sorted(glob.glob(pattern)))

        unique = []
        seen = set()
        for port in ports:
            key = os.path.realpath(port)
            if key not in seen:
                unique.append(port)
                seen.add(key)
        return unique

    def _is_connected(self):
        return self.ser is not None and getattr(self.ser, "is_open", False)

    def _try_connect(self):
        if self._is_connected():
            return True

        self.fake = False
        candidates = self._candidate_ports()
        if not candidates:
            self.last_error = "no candidate serial ports found"
            if self.allow_fake:
                self.fake = True
                self.get_logger().warn("No tool serial ports found; running in FAKE mode")
            else:
                self.get_logger().error(
                    "No tool serial ports found. Expected /dev/serial/by-id/*, "
                    "/dev/ttyACM*, or /dev/ttyUSB*. Tool commands disabled."
                )
            return False

        errors = []
        for port in candidates:
            try:
                self.ser = serial.Serial(port, self.baud, timeout=1.0)
                self.port = port
                self.last_error = ""
                self.fake = False
                self.get_logger().info(f"Connected to Tool @ {self.port}")
                return True
            except (serial.SerialException, OSError) as exc:
                errors.append(f"{port}: {exc}")

        self.ser = None
        self.last_error = "; ".join(errors[-3:])
        if self.allow_fake:
            self.fake = True
            self.get_logger().warn(f"Could not open a tool serial port, running in FAKE mode: {self.last_error}")
        else:
            self.get_logger().error(
                f"Could not open any tool serial port; fake mode is disabled. {self.last_error}"
            )
        return False

    def _publish_status(self):
        status = {
            "connected": self._is_connected(),
            "fake": self.fake,
            "port": self.port,
            "configured_port": self.configured_port,
            "baud": self.baud,
            "last_error": self.last_error,
            "current_cmd": self.current_cmd,
            "command_seq": self.command_seq,
            "request_seq": self.request_seq,
            "last_requested_cmd": self.last_requested_cmd,
            "last_command_ok": self.last_command_ok,
        }
        self.status_pub.publish(String(data=json.dumps(status)))

    def _connection_tick(self):
        if not self._is_connected() and not self.fake:
            self._try_connect()
        self._publish_status()

    def handle_cmd(self, msg: Int8):
        cmd = msg.data
        
        # Validating allowed commands: Now including 2 and 3
        if cmd not in [1, 0, -1, 2, 3]:
            self.get_logger().warn(f"Received invalid command: {cmd}")
            return

        self.request_seq += 1
        self.last_requested_cmd = cmd
        self.last_command_ok = False
        
        # Map IDs to readable names for logging
        cmd_names = {
            1: "SCREW",
            0: "STOP",
            -1: "UNSCREW",
            2: "GRAB",
            3: "RELEASE"
        }
        
        # Send to Hardware
        if not self.fake and not self._is_connected():
            self._try_connect()

        if not self.fake and self._is_connected():
            try:
                # Protocol: send the number as string with newline
                payload = f"{cmd}\n".encode('utf-8')
                self.ser.write(payload)
                self.ser.flush()
                self.current_cmd = cmd
                self.command_seq += 1
                self.last_command_ok = True
                self.last_error = ""
                self.get_logger().info(f"Sent {cmd_names.get(cmd)} command: {cmd}")
            except Exception as e:
                self.get_logger().error(f"Serial Error: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
                self.last_error = str(e)
        else:
            if self.fake:
                self.current_cmd = cmd
                self.command_seq += 1
                self.last_command_ok = True
                self.get_logger().info(f"[FAKE MODE] Executing {cmd_names.get(cmd)}: {cmd}")
            else:
                self.get_logger().error(
                    f"Tool is not connected; rejected {cmd_names.get(cmd)} command: {cmd}"
                )
        self._publish_status()

def main(args=None):
    rclpy.init(args=args)
    node = ToolCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser: 
            try:
                node.ser.write(b"0\n")
            except Exception:
                pass
            try:
                node.ser.close()
            except Exception:
                pass
        try:
            node.destroy_node()
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == '__main__':
    main()

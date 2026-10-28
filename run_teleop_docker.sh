#!/usr/bin/env bash
set -euo pipefail

CLEAN_BUILD=0
for arg in "$@"; do
  if [[ "$arg" == "--clean" ]]; then
    CLEAN_BUILD=1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD=(docker compose)
  DOCKER_COMPOSE_ENV=()
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD=(docker-compose)
  # Legacy docker-compose is a Python app and can break if user-site packages
  # override the distro-pinned docker SDK dependencies.
  DOCKER_COMPOSE_ENV=(env PYTHONNOUSERSITE=1 PYTHONPATH=)
else
  echo "Docker Compose is required but neither 'docker compose' nor 'docker-compose' was found in PATH." >&2
  exit 1
fi

if ! command -v xhost >/dev/null 2>&1; then
  echo "xhost is required for GUI forwarding but was not found in PATH." >&2
  exit 1
fi

xhost +local:root >/dev/null

# The entrypoint uses root only to stage Vuer TLS files, then switches to this
# UID/GID so ROS and build artifacts remain owned by the current host user.
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)
export RENDER_GID=$(getent group render | cut -d: -f3)
if [[ -z "${RENDER_GID}" ]]; then
  RENDER_GID=110
fi

exec "${DOCKER_COMPOSE_ENV[@]}" "${DOCKER_COMPOSE_CMD[@]}" run --rm teleop bash -lc "
  set -e
  cd /ws

  CLEAN_BUILD=${CLEAN_BUILD}

  runtime_dependencies_ready() {
    source /opt/ros/humble/setup.bash
    python3 -c 'import serial, pymodbus, vuer' >/dev/null 2>&1 || return 1
    ros2 pkg prefix orbbec_camera >/dev/null 2>&1 || return 1
  }

  workspace_ready() {
    # A host-side --symlink-install can look complete from the host while its
    # absolute links are dangling in the container, where the workspace lives
    # at /ws.  Validate package discovery from the container instead of relying
    # on one or two files being present.
    (
      set +u
      [ -f /ws/install/setup.bash ] || exit 1
      source /opt/ros/humble/setup.bash
      source /ws/install/setup.bash >/dev/null 2>&1 || exit 1

      for package in \
        dual_arm_scene_description \
        dual_arm_moveit_config \
        arm_teleop \
        vision_agent \
        disassembly_skill \
        tool_controller \
        tool_camera_pkg \
        robotiq_ft_sensor_hardware; do
        ros2 pkg prefix \"\${package}\" >/dev/null 2>&1 || exit 1
      done

      [ -x /ws/install/arm_teleop/lib/arm_teleop/vuer_quest_bridge ] || exit 1
      [ -x /ws/install/disassembly_skill/lib/disassembly_skill/laptop_auto_agent ] || exit 1
      [ -e /ws/install/dual_arm_moveit_config/share/dual_arm_moveit_config/local_setup.bash ] || exit 1
    )
  }

  build_complete_stack() {
    echo '[run_teleop] Building the complete teleoperation and disassembly stack...'
    source /opt/ros/humble/setup.bash
    colcon build --packages-up-to \
      dual_arm_moveit_config \
      arm_teleop \
      vision_agent \
      disassembly_skill \
      tool_controller \
      tool_camera_pkg \
      robotiq_ft_sensor_hardware \
      robotiq_ft_sensor_description \
      bio_ik \
      exotica_collision_scene_fcl_latest \
      exotica_core_task_maps \
      exotica_ik_solver \
      exotica_ompl_solver \
      exotica_python \
      --executor sequential
    source /ws/install/setup.bash
  }

  if ! runtime_dependencies_ready; then
    echo '[run_teleop] The Docker image is missing a required runtime dependency.' >&2
    echo '[run_teleop] Rebuild it with: env PYTHONNOUSERSITE=1 PYTHONPATH= docker-compose build teleop' >&2
    exit 3
  elif [ \"\${CLEAN_BUILD}\" = \"1\" ]; then
    echo '[run_teleop] --clean: removing build/, install/, log/ for a fresh colcon build.'
    rm -rf /ws/build /ws/install /ws/log
    build_complete_stack
  elif workspace_ready; then
    echo '[run_teleop] Complete container workspace is ready — skipping colcon build.'
    source /ws/install/setup.bash
  elif [ -d /ws/build ] || [ -d /ws/install ]; then
    echo >&2
    echo '[run_teleop] The existing build/install tree is incomplete or was built with host-absolute symlinks.' >&2
    echo '[run_teleop] Exit this container and run: ./run_teleop_docker.sh --clean' >&2
    exit 2
  else
    build_complete_stack
  fi

  exec bash
"

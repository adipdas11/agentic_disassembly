from inspect import signature

from dual_arm_moveit_config.exotica_planner import RemoteExoticaIKClient


def test_remote_client_accepts_prevalidated_readiness() -> None:
    parameter = signature(RemoteExoticaIKClient.__init__).parameters["skip_ready_wait"]

    assert parameter.default is False

"""
agents/arm_approach_agent.py - Arm Approach Agent node.

This adapter exposes agents.arm_approach as a commander-compatible agent.
It calculates the closest IK solution from the current base pose and
runs arm trajectories to reach the grasp target.
"""

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ArmApproachAgent:
    """Commander-facing wrapper around the arm approach runtime."""

    AGENT_NAME = "Arm Approach Agent"

    def __init__(self):
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true"

    async def execute(
        self,
        params: Dict[str, Any],
        context_id: str = "",
    ) -> Dict[str, Any]:
        params = dict(params or {})
        use_mock = _bool_param(params, "mock", self._use_mock)
        if use_mock:
            return await self._mock_execute(params)

        return await self._subprocess_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        logger.info("[%s] Mock: arm approach completed.", self.AGENT_NAME)
        return {
            "result": {
                "success": True,
                "status_code": "ARM_APPROACH_SUCCESS",
                "phase": "mock",
                "message": "Mock arm approach completed.",
                "next_agent": None,
                "nav_result": {
                    "success": True,
                    "skipped": True,
                    "phase": "mock",
                },
                "arm_motion_skipped": False,
            },
            "success": True,
        }

    async def _subprocess_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        timeout_sec = float(os.getenv("APPROACH_AGENT_TIMEOUT_SEC", "420"))
        cmd = _approach_subprocess_cmd(context_id)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(params, ensure_ascii=False).encode("utf-8")),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return _fail_result(
                "timeout",
                f"ArmApproachAgent subprocess timed out after {timeout_sec:.1f}s.",
            )

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            logger.error("[%s] subprocess failed: %s", self.AGENT_NAME, stderr_text)
            return _fail_result("subprocess_failed", stderr_text or "arm approach subprocess failed")
        try:
            payload = json.loads(stdout_text or "{}")
        except json.JSONDecodeError as exc:
            logger.error("[%s] invalid subprocess JSON stdout: %s", self.AGENT_NAME, stdout_text)
            return _fail_result(
                "invalid_subprocess_json",
                f"{exc}; stderr={stderr_text[-1000:]}",
            )
        if not isinstance(payload, dict):
            return _fail_result("invalid_subprocess_payload", "arm approach subprocess returned non-object JSON")
        if stderr_text:
            logger.info("[%s] subprocess log:\n%s", self.AGENT_NAME, stderr_text[-4000:])
        if "result" not in payload:
            return {
                "result": payload,
                "success": bool(payload.get("success", False)),
            }
        return payload


def _bool_param(params: Dict[str, Any], name: str, default: bool) -> bool:
    raw_value = params.get(name)
    if raw_value is None:
        return bool(default)
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() not in {"0", "false", "no", "off", ""}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _approach_python_bin() -> str:
    return (
        os.getenv("APPROACH_AGENT_PYTHON_BIN")
        or os.getenv("ROS_PYTHON_BIN")
        or "/usr/bin/python3"
    )


def _ros_setup_scripts() -> list[str]:
    return [
        os.getenv("ROS_SETUP_BASH", "/opt/ros/humble/setup.bash"),
        os.getenv("ROS_OVERLAY_SETUP_BASH", "/workspaces/install/setup.bash"),
    ]


def _approach_subprocess_cmd(context_id: str) -> str:
    parts = ["unset VIRTUAL_ENV PYTHONHOME PYTHONPATH"]
    parts.append("export PYTHONDONTWRITEBYTECODE=1")
    for setup_script in _ros_setup_scripts():
        if setup_script:
            quoted = shlex.quote(setup_script)
            parts.append(f"if [ -f {quoted} ]; then source {quoted}; fi")
    parts.append(f"export ROS_DOMAIN_ID={shlex.quote(os.getenv('ROS_DOMAIN_ID', '1'))}")
    parts.append(f"cd {shlex.quote(str(_repo_root()))}")
    parts.append(
        "exec "
        f"{shlex.quote(_approach_python_bin())} "
        "-m agents.arm_approach.subprocess_entry "
        f"--context-id {shlex.quote(context_id)}"
    )
    return " && ".join(parts)


def _fail_result(phase: str, message: str) -> Dict[str, Any]:
    return {
        "result": {
            "success": False,
            "status_code": "ARM_APPROACH_FAIL",
            "phase": phase,
            "error": message,
            "next_agent": None,
        },
        "success": False,
    }

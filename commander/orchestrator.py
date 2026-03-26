import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

import httpx
from langgraph.graph import END, StateGraph

from .brain import Brain
from .logger import TraceLogger
from .state import CommanderState

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Builds and runs the LangGraph decision graph.

    Graph topology:
        observe_node → reason_node
            reason_node --[nav_agent]-------> nav_node ┐
            reason_node --[grasp_agent]-----> grasp_node ├─→ update_memory_node → observe_node
            reason_node --[approach_agent]--> approach_node ┘
            reason_node --[view_agent]------> view_node ┘
            reason_node --[DONE]-----------> END
    """

    def __init__(self, trace_logger: TraceLogger, use_mock: bool = True):
        self.logger = trace_logger
        self.brain = Brain(use_mock=use_mock)
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.use_mock = use_mock
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommanderState)

        # Core nodes
        workflow.add_node("input_node", self._input_node)
        workflow.add_node("find_node", self._find_node)
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        # Individual Agent nodes (A2A Clients)
        workflow.add_node("nav_node", self._nav_node)
        workflow.add_node("nav_move_node", self._nav_move_node)
        workflow.add_node("grasp_node", self._grasp_node)
        workflow.add_node("approach_node", self._approach_node)
        workflow.add_node("view_node", self._view_node)
        workflow.add_node("get_item_info_node", self._get_item_info_node)

        # Entry point
        workflow.set_entry_point("input_node")

        # Fixed edges
        workflow.add_edge("input_node", "find_node")
        workflow.add_edge("get_item_info_node", "nav_move_node")
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_node", "nav_move_node")
        workflow.add_edge("grasp_node", "update_memory_node")
        workflow.add_edge("approach_node", "update_memory_node")
        workflow.add_edge("view_node", "update_memory_node")
        workflow.add_edge("update_memory_node", "observe_node")

        # find_node → get_item_info_node or END
        workflow.add_conditional_edges(
            "find_node",
            self._route_find,
            {"get_item_info_node": "get_item_info_node", "end": END},
        )

        # reason_node → agent node or END
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {
                "nav_node":      "nav_node",
                "grasp_node":    "grasp_node",
                "approach_node": "approach_node",
                "view_node":     "view_node",
                "end":           END,
            },
        )
        workflow.add_conditional_edges(
            "nav_move_node",
            self._route_nav_move,
            {
                "observe_node": "observe_node",
                "update_memory_node": "update_memory_node",
            },
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: human input (runs ONCE at graph start)
    # ------------------------------------------------------------------

    async def _input_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Display available objects from config/objects.yaml and wait for user selection.
        If task_description is already set (e.g. injected by tests), skip stdin.
        """
        # Allow tests or programmatic callers to pre-fill task_description
        existing = state.get("task_description", "").strip()
        if existing:
            logger.info(f"[input_node] Task pre-filled: {existing}")
            return {"task_description": existing, "current_status": "INPUT_RECEIVED"}

        from agents.find_agent import _load_objects
        objects = _load_objects()

        print("\n" + "=" * 60)
        print("  VLM-RL 多代理人抓取系統")
        print("=" * 60)
        print("請選擇你要抓取的目標物：")
        for i, obj in enumerate(objects, 1):
            print(f"  {i}. {obj.get('label', obj.get('id', '未知'))}")
        print("=" * 60)

        loop = asyncio.get_event_loop()
        while True:
            choice_str = await loop.run_in_executor(
                None,
                lambda: input(f"\n請輸入目標物編號 (1-{len(objects)})：\n> "),
            )

            try:
                idx = int(choice_str.strip()) - 1
                if 0 <= idx < len(objects):
                    selected = objects[idx]
                    break
                else:
                    print("❌ 錯誤：輸入數字不在範圍內，請重新輸入。")
            except ValueError:
                print("❌ 錯誤：格式不正確，請輸入有效數字。")

        task_desc = f"抓取{selected.get('label', selected.get('id', '目標物'))}"

        task_desc = task_desc.strip() or "請抓取桌上的目標物件"
        logger.info(f"[input_node] Task received: {task_desc}")
        print(f"\n✅ 任務已確認：{task_desc}")
        print("-" * 60)

        return {
            "task_description": task_desc,
            "target_object": {
                "id": selected.get("id"),
                "label": selected.get("label")
            },
            "current_status": "INPUT_RECEIVED",
        }

    # ------------------------------------------------------------------
    # Node: find (runs once — YOLO detection + human confirmation)
    # ------------------------------------------------------------------

    async def _find_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        1. Call FindAgent to run YOLO on all camera images (returns annotated images + metadata).
        2. Save annotated images to logs/find_candidates/.
        3. Print detection list, wait for user to pick a number or type 'no'.
        """
        task_desc = state.get("task_description", "")

        target_obj = state.get("target_object", {})

        from agents.find_agent import FindAgent
        agent = FindAgent(http_client=self.http_client)
        result = await agent.execute({
            "task_description": task_desc,
            "target_object": target_obj
        })
        agent_result = result.get("result", {})
        yolo_detections: Dict[str, Any] = agent_result.get("yolo_detections", {})
        annotated_images: Dict[str, str] = agent_result.get("annotated_images", {})

        # Save individual annotated full images if server returned them
        if annotated_images:
            import base64
            save_dir = Path("logs/find_candidates")
            save_dir.mkdir(parents=True, exist_ok=True)
            for det_id_str, b64_img in annotated_images.items():
                det_info = yolo_detections.get(det_id_str) or yolo_detections.get(int(det_id_str), {})
                cam_name = det_info.get("camera", "unknown")
                img_path = save_dir / f"{det_id_str}_{cam_name}.jpg"
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(b64_img))

        # Print detections for user
        print("\n" + "=" * 55)
        print("  🔍 YOLO 偵測結果：")
        print("=" * 55)
        if not yolo_detections:
            print("  ⚠ 找不到任何物品。")
        else:
            for det_id, det in yolo_detections.items():
                print(f"  [{det_id}] {det.get('label','?')}  信心度={det.get('conf', 0):.0%}  相機={det.get('camera','?')}")
            print(f"\n  📁 有匡到的相片已存至: logs/find_candidates/")
        print("=" * 55)

        loop = asyncio.get_event_loop()
        
        if not yolo_detections:
            # If no detections, just pause and exit
            await loop.run_in_executor(
                None,
                lambda: input("按下 Enter 鍵結束任務..."),
            )
            logger.info("[find_node] No target found. User acknowledged.")
            return {
                "yolo_detections": yolo_detections,
                "selected_detection_id": 0,
                "find_complete": True,
                "current_status": "TARGET_NOT_FOUND",
            }

        while True:
            choice_str = await loop.run_in_executor(
                None,
                lambda: input(
                    f"\n請輸入目標物編號 (1-{len(yolo_detections)}) 或輸入 no 表示未找到：\n> "
                ),
            )

            choice = choice_str.strip().lower()
            if choice == "no":
                logger.info("[find_node] User indicated no valid target found.")
                return {
                    "yolo_detections": yolo_detections,
                    "selected_detection_id": 0,
                    "find_complete": True,
                    "current_status": "TARGET_NOT_FOUND",
                }

            try:
                selected_id = int(choice)
                if selected_id in yolo_detections or str(selected_id) in yolo_detections:
                    # Depending on how the dict keys were parsed (int or str)
                    selected_id = selected_id if selected_id in yolo_detections else str(selected_id)
                    det = yolo_detections[selected_id]
                    break
                else:
                    print("❌ 錯誤：輸入號碼不在選項內，請重新輸入。")
            except ValueError:
                print("❌ 錯誤：格式不正確，請輸入數字或 'no'。")

        label = det.get("label", "目標物")
        logger.info(f"[find_node] User selected detection {selected_id}: {label}")
        print(f"\n✅ 選定目標：[{selected_id}] {label}")
        print("-" * 55)

        return {
            "yolo_detections": yolo_detections,
            "selected_detection_id": selected_id,
            "find_complete": True,
            "current_status": "TARGET_FOUND",
        }

    # ------------------------------------------------------------------
    # Conditional edge: after find_node
    # ------------------------------------------------------------------

    def _route_find(self, state: CommanderState) -> str:
        """Route to get_item_info_node if target found, else END."""
        if state.get("selected_detection_id", 0) == 0:
            logger.info("[route_find] No target → ending graph.")
            return "end"
        return "get_item_info_node"

    # ------------------------------------------------------------------
    # Node: get_item_info (runs once — retrieve full 3D info)
    # ------------------------------------------------------------------

    async def _get_item_info_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Takes the user-selected detection, calls GetItemInfoAgent on 3090 via A2A
        with all available fixed-room RGB images to retrieve 3D
        position, size, and other properties.
        Stores result in state["target_object"] for the rest of the session.
        """
        from agents.get_item_info_agent import GetItemInfoAgent
        from commander.camera import get_camera_image_base64
        from commander.camera_groups import configured_room_cameras
        
        agent = GetItemInfoAgent(http_client=self.http_client)
        
        # Get the actual ID defined in objects.yaml
        target_obj = state.get("target_object", {})
        yolo_class = target_obj.get("id", "unknown")

        selected_detection_id = state.get("selected_detection_id", 0)
        yolo_detections = state.get("yolo_detections", {})
        selected_det = (
            yolo_detections.get(selected_detection_id)
            or yolo_detections.get(str(selected_detection_id))
            or {}
        )
        selected_camera = str(selected_det.get("camera", "")).strip()
        room_cameras = configured_room_cameras()
        primary_camera = selected_camera if selected_camera in room_cameras else ""

        if len(room_cameras) < 2:
            logger.error(
                "[get_item_info_node] Not enough configured room cameras for triangulation. detection=%s camera=%s configured=%s",
                selected_detection_id,
                selected_camera,
                room_cameras,
            )
            print("❌ 失敗: 可用的房間固定相機少於 2 台，無法進行 3D 定位。")
            return {"current_status": "ITEM_INFO_FAILED"}

        if selected_camera and not primary_camera:
            logger.warning(
                "[get_item_info_node] Selected camera %s is not a configured room camera; falling back to multi-view auto selection.",
                selected_camera,
            )

        print(f"\n📷 正在擷取全部房間相機 RGB 影像 ({', '.join(room_cameras)})...")
        image_results = await asyncio.gather(
            *(get_camera_image_base64(camera_name, timeout_sec=15.0) for camera_name in room_cameras)
        )
        camera_images = {
            camera_name: image_b64
            for camera_name, image_b64 in zip(room_cameras, image_results)
            if image_b64
        }
        available_cameras = [camera_name for camera_name in room_cameras if camera_images.get(camera_name)]
        missing_cameras = [
            camera_name
            for camera_name, image_b64 in zip(room_cameras, image_results)
            if not image_b64
        ]

        if len(available_cameras) < 2:
            logger.error(
                "[get_item_info_node] Not enough captured room camera images. selected=%s available=%s missing=%s",
                selected_camera,
                available_cameras,
                missing_cameras,
            )
            print(
                f"❌ 失敗: 成功收到的房間相機影像不足 2 張。"
                f" available={available_cameras}, missing={missing_cameras}"
            )
            return {"current_status": "ITEM_INFO_FAILED"}

        if missing_cameras:
            logger.warning(
                "[get_item_info_node] Some room cameras did not return images; continuing with available=%s missing=%s",
                available_cameras,
                missing_cameras,
            )
            print(f"⚠ 部分房間相機未回圖，改用已收到的相機: {available_cameras}")

        if primary_camera and primary_camera not in camera_images:
            logger.warning(
                "[get_item_info_node] Selected primary camera %s did not return an image; clearing primary camera hint.",
                primary_camera,
            )
            primary_camera = ""

        print(
            f"📡 傳送目標 '{yolo_class}' 與全部可用房間相機 "
            f"{available_cameras} 至 3090 A2A Server 分析 3D 姿態..."
        )
        params = {
            "yolo_class": yolo_class,
            "selected_camera": primary_camera,
            "camera_names": available_cameras,
            "camera_images": {
                camera_name: camera_images[camera_name]
                for camera_name in available_cameras
            },
        }
        result = await agent.execute(params)
        target_object = {
            **target_obj,
            **(result.get("result", {}) or {}),
        }

        pos = target_object.get("center_world", "N/A")
        logger.info(f"[get_item_info_node] Target '{yolo_class}' 3D info retrieved: {pos}")
        print(f"\n📦 目標物立體資訊已取得！ 3D 中心點 = {pos}")
        
        current_rank = int(state.get("current_goal_rank", 1) or 1)
        if current_rank < 1:
            current_rank = 1

        return {
            "target_object": target_object,
            "current_goal_rank": current_rank,
            "nav_move_source": "bootstrap",
            "force_initialpose": False,
            "current_status": "ITEM_INFO_READY",
        }

    # ------------------------------------------------------------------
    # Node: observe
    # ------------------------------------------------------------------


    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        """Fetch current environment observation (mock or Rosbridge)."""
        task_desc = state.get("task_description", "抓取目標物件")

        if self.use_mock:
            obs = {
                "description": (
                    f"任務：{task_desc} "
                    f"| Mock scene at step {state.get('retry_count', 0)}: "
                    "Target object visible on table with partial occlusion."
                ),
                "image_base64": None,
            }
        else:
            from .camera import get_camera_image_base64
            image_b64 = await get_camera_image_base64("Camera_Car", timeout_sec=15.0)
            obs = {
                "description": task_desc,
                "image_base64": image_b64,
            }

        logger.info(f"[observe_node] Observation captured (Has Image: {bool(obs.get('image_base64'))}).")
        return {"current_observation": obs, "current_status": "OBSERVED"}

    # ------------------------------------------------------------------
    # Node: reason
    # ------------------------------------------------------------------

    async def _reason_node(self, state: CommanderState) -> Dict[str, Any]:
        """Call the VLM Brain to produce a BrainDecision."""
        out = await self.brain.reason(state)
        decision = out["prediction"]
        logger.info(
            f"[reason_node] call_module={decision.call_module} "
            f"latency={out['latency']:.2f}s"
        )
        return {
            "reasoning": decision.reasoning,
            "call_module": decision.call_module,
            "module_params": decision.module_params,
            "decision_latency": out["latency"],
            "current_status": "REASONED",
        }

    # ------------------------------------------------------------------
    # Conditional edge
    # ------------------------------------------------------------------

    def _route_decision(
        self, state: CommanderState
    ) -> Literal["nav_node", "grasp_node", "approach_node", "view_node", "end"]:
        module = state.get("call_module", "")
        if module == "DONE" or state.get("task_complete", False):
            logger.info("[route] Task complete — ending graph.")
            return "end"
        mapping = {
            "nav_agent":      "nav_node",
            "grasp_agent":    "grasp_node",
            "approach_agent": "approach_node",
            "view_agent":     "view_node",
        }
        return mapping.get(module, "end")

    def _route_nav_move(self, state: CommanderState) -> Literal["observe_node", "update_memory_node"]:
        source = state.get("nav_move_source", "")
        if source == "bootstrap":
            return "observe_node"
        return "update_memory_node"

    # ------------------------------------------------------------------
    # Agent nodes (each is an A2A Client calling RTX 3090)
    # ------------------------------------------------------------------

    async def _nav_node(self, state: CommanderState) -> Dict[str, Any]:
        """Prepare navigation context and delegate execution to nav_move_node."""
        current_rank = int(state.get("current_goal_rank", 1) or 1)
        if current_rank < 1:
            current_rank = 1
        goal_data, err = self._goal_pose_for_rank(state.get("target_object", {}), current_rank)
        update: Dict[str, Any] = {
            "current_goal_rank": current_rank,
            "nav_move_source": "reason_loop",
            "force_initialpose": bool(state.get("module_params", {}).get("force_initialpose", False)),
            "current_status": "NAV_CONTEXT_READY",
        }
        if err:
            update["agent_result"] = f"[NAV_CONTEXT_ERROR] {err}"
            update["agent_success"] = False
            update["nav_goal_pose"] = {}
        else:
            update["nav_goal_pose"] = goal_data
        return update

    async def _nav_move_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Blocking navigation executor:
        1. Warm-up publish /goal_pose for discovery.
        2. Request a global path from planner_server.
        3. Follow /received_global_plan with the pros-style discrete wheel controller.
        """
        source = state.get("nav_move_source", "reason_loop")
        target_object = state.get("target_object", {})
        group_ranking = target_object.get("group_ranking", []) or []
        if not group_ranking:
            return {
                "agent_result": "[NAV] group_ranking is empty, cannot navigate.",
                "agent_success": False,
                "nav_plan_ready": False,
                "nav_arrived": False,
                "nav_move_events": [],
                "current_status": "NAV_FAILED",
                "_exec_latency": 0.0,
            }

        start_t = time.time()
        same_rank_retries = max(0, int(os.getenv("NAV_SAME_RANK_RETRIES", "0")))
        max_attempt_per_rank = same_rank_retries + 1
        plan_timeout = float(os.getenv("NAV_PLAN_TIMEOUT_SEC", "8"))
        arrival_timeout = float(os.getenv("NAV_ARRIVAL_TIMEOUT_SEC", "120"))
        publish_interval = float(os.getenv("NAV_PUBLISH_INTERVAL_SEC", "0.5"))
        replan_period = float(os.getenv("NAV_REPLAN_PERIOD_SEC", "1.5"))
        follow_control_hz = float(os.getenv("NAV_FOLLOW_CONTROL_HZ", "10"))
        goal_tolerance_m = float(os.getenv("NAV_GOAL_TOLERANCE_M", "0.08"))
        goal_heading_tolerance_deg = float(
            os.getenv("NAV_GOAL_HEADING_TOLERANCE_DEG", "5.0")
        )
        path_target_distance_m = float(os.getenv("NAV_PATH_TARGET_DISTANCE_M", "0.5"))

        rank = int(state.get("current_goal_rank", 1) or 1)
        if rank < 1:
            rank = 1

        force_initialpose = bool(state.get("force_initialpose", False))
        all_events = []
        last_error = "unknown navigation error"
        last_attempt = 0

        while rank <= len(group_ranking):
            goal_data, goal_err = self._goal_pose_for_rank(target_object, rank)
            if goal_err:
                all_events.append({
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": rank + 1,
                    "detail": goal_err,
                })
                rank += 1
                continue

            for attempt in range(1, max_attempt_per_rank + 1):
                last_attempt = attempt
                publish_initialpose = force_initialpose
                if self.use_mock:
                    await asyncio.sleep(0.2)
                    mock_events = [
                        {"event": "goal_publishing", "rank": rank, "attempt": attempt},
                        {"event": "plan_ready", "rank": rank, "attempt": attempt},
                        {"event": "arrived", "rank": rank, "attempt": attempt},
                    ]
                    return {
                        "current_goal_rank": rank,
                        "nav_attempt": attempt,
                        "nav_goal_pose": goal_data,
                        "nav_plan_ready": True,
                        "nav_arrived": True,
                        "nav_move_events": mock_events,
                        "agent_result": f"[MOCK_NAV] Arrived at rank {rank} (attempt {attempt}).",
                        "agent_success": True,
                        "current_status": "NAV_COMPLETED",
                        "_exec_latency": time.time() - start_t,
                    }

                payload = {
                    "goal_pose": goal_data,
                    "publish_initialpose": publish_initialpose,
                    "initial_pose": self._default_initial_pose(),
                    "plan_timeout_sec": plan_timeout,
                    "arrival_timeout_sec": arrival_timeout,
                    "publish_interval_sec": publish_interval,
                    "replan_period_sec": replan_period,
                    "follow_control_hz": follow_control_hz,
                    "goal_tolerance_m": goal_tolerance_m,
                    "goal_heading_tolerance_deg": goal_heading_tolerance_deg,
                    "path_target_distance_m": path_target_distance_m,
                    "status_topic": "/nav_move/status",
                    "attempt": attempt,
                    "rank": rank,
                    "source": source,
                }
                result = await self._run_nav_move_runner(payload)
                events = result.get("events", [])
                all_events.extend(events)
                plan_ready = bool(result.get("plan_ready", False))
                success = bool(result.get("success", False))
                if publish_initialpose and plan_ready:
                    force_initialpose = False

                if success:
                    return {
                        "current_goal_rank": rank,
                        "nav_attempt": attempt,
                        "nav_goal_pose": goal_data,
                        "nav_plan_ready": plan_ready,
                        "nav_arrived": True,
                        "nav_move_events": all_events,
                        "agent_result": result.get(
                            "message",
                            f"[NAV] Arrived at rank {rank} (attempt {attempt}).",
                        ),
                        "agent_success": True,
                        "current_status": "NAV_COMPLETED",
                        "_exec_latency": time.time() - start_t,
                    }

                last_error = result.get("message", "navigation attempt failed")
                all_events.append(
                    {
                        "event": "attempt_failed",
                        "rank": rank,
                        "attempt": attempt,
                        "detail": last_error,
                    }
                )

            all_events.append(
                {
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": rank + 1,
                    "detail": "exhausted retries on current rank",
                }
            )
            rank += 1

        all_events.append(
            {
                "event": "navigation_failed",
                "detail": last_error,
            }
        )
        logger.error(f"[nav_move_node] Navigation failed completely. Inner error: {last_error}")
        return {
            "current_goal_rank": rank,
            "nav_attempt": last_attempt,
            "nav_goal_pose": {},
            "nav_plan_ready": False,
            "nav_arrived": False,
            "nav_move_events": all_events,
            "agent_result": f"[NAV] Failed after exhausting all ranks: {last_error}",
            "agent_success": False,
            "current_status": "NAV_FAILED",
            "_exec_latency": time.time() - start_t,
        }

    async def _grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        """GraspGen Agent Node: capture Camera_Car RGBD and call INF_GRASP."""
        from agents.grasp_agent import GraspAgent
        from commander.camera import get_camera_rgbd_base64

        target_object = state.get("target_object", {}) or {}
        object_id = target_object.get("id")
        if not object_id:
            logger.error("[grasp_node] target_object.id is missing.")
            return {
                "agent_result": "[GRASP] target_object.id is missing, cannot call grasp agent.",
                "agent_success": False,
                "current_status": "EXECUTED",
                "_exec_latency": 0.0,
            }

        print("\n📷 正在擷取 Camera_Car 的 RGBD 影像...")
        rgbd = await get_camera_rgbd_base64("Camera_Car", timeout_sec=15.0)
        if not rgbd:
            logger.error("[grasp_node] Failed to get Camera_Car RGBD image.")
            return {
                "agent_result": "[GRASP] Failed to capture Camera_Car RGBD image.",
                "agent_success": False,
                "current_status": "EXECUTED",
                "_exec_latency": 0.0,
            }

        params = dict(state.get("module_params", {}) or {})
        params.update(
            {
                "object_id": object_id,
                "camera_name": "Camera_Car",
                "rgb_base64": rgbd.get("rgb_base64"),
                "depth_base64": rgbd.get("depth_base64"),
            }
        )
        logger.info("[grasp_node] Captured Camera_Car RGBD and prepared grasp request for '%s'.", object_id)
        agent = GraspAgent(http_client=self.http_client)
        return await self._run_agent(agent, state, params_override=params)

    async def _approach_node(self, state: CommanderState) -> Dict[str, Any]:
        """Approach Agent Node: guide arm to pre-grasp point (local control, no GPU)."""
        from agents.approach_agent import ApproachAgent
        agent = ApproachAgent()
        return await self._run_agent(agent, state, has_http=False)

    async def _view_node(self, state: CommanderState) -> Dict[str, Any]:
        """View Agent Node: adjust camera/arm posture via INF_VIEW (A2A Server)."""
        from agents.view_agent import ViewAgent
        agent = ViewAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _run_agent(
        self,
        agent: Any,
        state: CommanderState,
        has_http: bool = True,
        params_override: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Common runner: call agent.execute() and return state updates."""
        params = params_override if params_override is not None else state.get("module_params", {})
        context_id = state.get("context_id", uuid.uuid4().hex)
        start = time.time()

        result = await agent.execute(params, context_id)
        exec_latency = time.time() - start

        logger.info(
            f"[{agent.AGENT_NAME}] finished in {exec_latency:.2f}s, "
            f"success={result['success']}"
        )
        return {
            "agent_result": result["result"],
            "agent_success": bool(result.get("success", False)),
            "current_status": "EXECUTED",
            "_exec_latency": exec_latency,
        }

    # ------------------------------------------------------------------
    # Node: update memory
    # ------------------------------------------------------------------

    async def _update_memory_node(self, state: CommanderState) -> Dict[str, Any]:
        """Append last action to history (capped at 3) and write trace log."""
        module = state.get("call_module", "")
        reasoning = state.get("reasoning", "")
        result = state.get("agent_result", "")
        decision_latency = state.get("decision_latency", 0.0)
        exec_latency = state.get("_exec_latency", 0.0)
        context_id = state.get("context_id", "")

        mem_entry = {
            "action": module,
            "reasoning": reasoning,
            "result": result,
            "success": bool(state.get("agent_success", bool(result))),
        }

        success_flag = bool(state.get("agent_success", bool(result)))
        self.logger.log_trace(
            agent_called=module,
            reasoning=reasoning,
            decision_latency=decision_latency,
            execution_latency=exec_latency,
            success=success_flag,
            context_id=context_id,
            extra_info={"result": result},
        )

        logger.info("[update_memory_node] History updated and trace logged.")
        return {
            "history_buffer": [mem_entry],
            "retry_count": state.get("retry_count", 0) + 1,
            "current_status": "MEMORY_UPDATED",
        }

    @staticmethod
    def _target_center_world_to_map_xy(target_object: Dict[str, Any]) -> tuple[float | None, float | None]:
        center_world = target_object.get("center_world", [])
        if not isinstance(center_world, list) or len(center_world) < 3:
            return None, None

        target_map_x = 6.0 - float(center_world[2])
        target_map_y = float(center_world[0]) - 3.314
        return target_map_x, target_map_y

    def _goal_pose_for_rank(self, target_object: Dict[str, Any], rank: int) -> tuple[Dict[str, Any], str]:
        group_ranking = target_object.get("group_ranking", []) or []
        rank_idx = rank - 1
        if rank_idx < 0 or rank_idx >= len(group_ranking):
            return {}, f"rank={rank} out of range"

        goal_data = group_ranking[rank_idx] or {}
        goal_pose_ros = goal_data.get("best_goal_pose_ros_map", [])
        if not isinstance(goal_pose_ros, list) or len(goal_pose_ros) < 2:
            return {}, f"rank={rank} missing best_goal_pose_ros_map"

        goal_x = float(goal_pose_ros[0])
        goal_y = float(goal_pose_ros[1])
        yaw = 0.0
        target_map_x, target_map_y = self._target_center_world_to_map_xy(target_object)
        if target_map_x is not None and target_map_y is not None:
            dx = target_map_x - goal_x
            dy = target_map_y - goal_y
            yaw = math.atan2(dy, dx)

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_pose = {
            "x": goal_x,
            "y": goal_y,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": qz,
            "qw": qw,
            "yaw": yaw,
        }
        if target_map_x is not None and target_map_y is not None:
            goal_pose["face_target_x"] = target_map_x
            goal_pose["face_target_y"] = target_map_y
        return goal_pose, ""

    def _default_initial_pose(self) -> Dict[str, Any]:
        return {
            "x": 3.112286942135556,
            "y": -3.2084078403508274,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": -0.016265232678593755,
            "qw": 0.9998677123529448,
            "covariance": [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892060437211,
            ],
        }

    async def _run_nav_move_runner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import shlex
        safe_payload = shlex.quote(json.dumps(payload))
        # Unset uv env vars so /usr/bin/python3 (3.10) runs cleanly with ROS 2.
        # Explicitly set PYTHONPATH and LD_LIBRARY_PATH to the ROS humble paths so
        # both the Python modules and their compiled .so extensions are found.
        ros_py = "/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
        ros_lib = "/opt/ros/humble/lib"
        cmd = (
            "unset VIRTUAL_ENV PYTHONPATH PYTHONHOME && "
            "source /opt/ros/humble/setup.bash && "
            "source /workspaces/nav_install/setup.bash 2>/dev/null || true && "
            f"export LD_LIBRARY_PATH={ros_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"PYTHONPATH={ros_py} "
            f"/usr/bin/python3 -m commander.nav_move_runner --payload {safe_payload}"
        )

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable='/bin/bash'
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode().strip() or "nav_move_runner failed"
            logger.error(f"[nav_move_node] runner exit={proc.returncode}: {err_msg}")
            return {
                "success": False,
                "plan_ready": False,
                "message": err_msg,
                "events": [],
            }

        try:
            return json.loads(stdout.decode().strip() or "{}")
        except json.JSONDecodeError:
            text = stdout.decode().strip()
            logger.error(f"[nav_move_node] runner returned non-JSON output: {text}")
            return {
                "success": False,
                "plan_ready": False,
                "message": "Invalid nav_move_runner output",
                "events": [],
            }

    async def aclose(self) -> None:
        """Clean up the shared httpx client."""
        await self.http_client.aclose()

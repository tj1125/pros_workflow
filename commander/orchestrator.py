import asyncio
import logging
import os
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
        self.http_client = httpx.AsyncClient(timeout=30.0)
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
        workflow.add_node("grasp_node", self._grasp_node)
        workflow.add_node("approach_node", self._approach_node)
        workflow.add_node("view_node", self._view_node)
        workflow.add_node("get_item_info_node", self._get_item_info_node)

        # Entry point
        workflow.set_entry_point("input_node")

        # Fixed edges
        workflow.add_edge("input_node", "find_node")
        workflow.add_edge("get_item_info_node", "observe_node")
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_node", "update_memory_node")
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
        to retrieve 3D position, size, and other properties.
        Stores result in state["target_object"] for the rest of the session.
        """
        det_id = state.get("selected_detection_id", 0)
        yolo_detections = state.get("yolo_detections", {})
        det = yolo_detections.get(det_id, {})

        from agents.get_item_info_agent import GetItemInfoAgent
        from commander.camera import get_camera_image_base64
        import asyncio
        
        agent = GetItemInfoAgent(http_client=self.http_client)
        
        # Get the actual ID defined in objects.yaml
        target_obj = state.get("target_object", {})
        yolo_class = target_obj.get("id", "unknown")

        print(f"\n📷 正在擷取立體相機影像 (Camera_Room1_1, Camera_Room1_2)...")
        # Fetch images in parallel
        cam_a_b64, cam_b_b64 = await asyncio.gather(
            get_camera_image_base64("Camera_Room1_1", timeout_sec=15.0),
            get_camera_image_base64("Camera_Room1_2", timeout_sec=15.0)
        )

        if not cam_a_b64 or not cam_b_b64:
            logger.error("[get_item_info_node] Failed to get camera images.")
            print("❌ 失敗: 未能取得兩個相機的影像。")
            return {"current_status": "ITEM_INFO_FAILED"}

        print(f"📡 傳送目標 '{yolo_class}' 資訊至 3090 A2A Server 分析 3D 姿態...")
        params = {
            "yolo_class": yolo_class,
            "cam_a_b64": cam_a_b64,
            "cam_b_b64": cam_b_b64,
        }
        result = await agent.execute(params)
        target_object = result.get("result", {})

        pos = target_object.get("center_world", "N/A")
        logger.info(f"[get_item_info_node] Target '{yolo_class}' 3D info retrieved: {pos}")
        print(f"\n📦 目標物立體資訊已取得！ 3D 中心點 = {pos}")
        
        # Determine current goal rank (starts at 1)
        current_rank = state.get("current_goal_rank", 1)
        
        # Publish initial and goal poses to ROS 2 non-blocking
        asyncio.create_task(self._publish_poses(target_object, current_rank))

        return {
            "target_object": target_object,
            "current_goal_rank": current_rank,
            "current_status": "ITEM_INFO_READY",
        }

    async def _publish_poses(self, target_object: Dict[str, Any], rank: int) -> None:
        """Asynchronously publishes /initialpose and /goal_pose 5 times using ROS 2."""
        import asyncio
        import json

        # Handle hardcoded initialpose
        initial_pose_msg = {
            "header": {
                "stamp": {
                    "sec": 1772448532,
                    "nsec": 582141336
                },
                "frame_id": "map"
            },
            "pose": {
                "pose": {
                    "position": {
                        "x": 3.9249487403129732,
                        "y": -3.3053666797849046,
                        "z": 0
                    },
                    "orientation": {
                        "x": 0,
                        "y": 0,
                        "z": -0.2743168365546609,
                        "w": 0.9616393675295554
                    }
                },
                "covariance": [
                    0.25, 0, 0, 0, 0, 0,
                    0, 0.25, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0.06853892060437211
                ]
            }
        }
        initial_pose_str = json.dumps(initial_pose_msg)
        
        # Handle goal_pose dynamically based on rank
        group_ranking = target_object.get("group_ranking", [])
        if not group_ranking:
            logger.warning("[publish_poses] No group_ranking found in target_object. Skipping goal_pose.")
            goal_pose_str = ""
        else:
            rank_idx = rank - 1
            if rank_idx < 0 or rank_idx >= len(group_ranking):
                logger.warning(f"[publish_poses] Rank {rank} out of bounds (0-{len(group_ranking)-1}). Using rank 1.")
                rank_idx = 0
            
            goal_data = group_ranking[rank_idx]
            # Assumes best_goal_pose_ros_map exists in the goal data from the server
            goal_pose_ros = goal_data.get("best_goal_pose_ros_map", {})
            
            if goal_pose_ros:
                goal_pose_msg = {
                    "header": {"frame_id": "map"},
                    "pose": {
                        "position": {
                            "x": goal_pose_ros.get("position", {}).get("x", 0.0),
                            "y": goal_pose_ros.get("position", {}).get("y", 0.0),
                            "z": goal_pose_ros.get("position", {}).get("z", 0.0)
                        },
                        "orientation": {
                            "x": goal_pose_ros.get("orientation", {}).get("x", 0.0),
                            "y": goal_pose_ros.get("orientation", {}).get("y", 0.0),
                            "z": goal_pose_ros.get("orientation", {}).get("z", 0.0),
                            "w": goal_pose_ros.get("orientation", {}).get("w", 1.0)
                        }
                    }
                }
                goal_pose_str = json.dumps(goal_pose_msg)
            else:
                logger.warning("[publish_poses] best_goal_pose_ros_map not found. Skipping goal_pose.")
                goal_pose_str = ""

        logger.info(f"[publish_poses] Publishing topics for Rank {rank}...")

        # Fire and forget subprocess loop
        script = f"""
        for i in {{1..5}}; do
            ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{initial_pose_str}' &
            """
        if goal_pose_str:
            script += f"ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped '{goal_pose_str}' &\n"
        
        script += """
            wait
            sleep 0.5
        done
        """
        
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"[publish_poses] Topic publish failed: {stderr.decode()}")
        else:
            logger.info(f"[publish_poses] Successfully published /initialpose & /goal_pose (Rank {rank}).")

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

    # ------------------------------------------------------------------
    # Agent nodes (each is an A2A Client calling RTX 3090)
    # ------------------------------------------------------------------

    async def _nav_node(self, state: CommanderState) -> Dict[str, Any]:
        """Nav Agent Node: navigate robot base via INF_NAV (A2A Server on RTX 3090)."""
        from agents.nav_agent import NavAgent
        agent = NavAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        """GraspGen Agent Node: generate 6-DoF grasp pose via INF_GRASP (A2A Server)."""
        from agents.grasp_agent import GraspAgent
        agent = GraspAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

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
    ) -> Dict[str, Any]:
        """Common runner: call agent.execute() and return state updates."""
        params = state.get("module_params", {})
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
            "success": bool(result),
        }

        self.logger.log_trace(
            agent_called=module,
            reasoning=reasoning,
            decision_latency=decision_latency,
            execution_latency=exec_latency,
            success=bool(result),
            context_id=context_id,
            extra_info={"result": result},
        )

        logger.info("[update_memory_node] History updated and trace logged.")
        return {
            "history_buffer": [mem_entry],
            "retry_count": state.get("retry_count", 0) + 1,
            "current_status": "MEMORY_UPDATED",
        }

    async def aclose(self) -> None:
        """Clean up the shared httpx client."""
        await self.http_client.aclose()

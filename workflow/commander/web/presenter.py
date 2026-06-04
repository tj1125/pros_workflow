from __future__ import annotations

from typing import Any, Dict

def _assistant_message_from_update(node_name: str, state_update: Dict[str, Any]) -> str:
    execution = state_update.get("last_execution", {}) or {}
    if node_name == "greeting_node" and state_update.get("current_status") == "GREETING_SENT":
        return str(execution.get("message", "") or "嗨～有什麼需要幫忙的嗎？")
    if node_name == "ai_reply_node":
        return str(state_update.get("ai_reply", "") or "")
    if node_name == "goodbye_node":
        return str(execution.get("message", "") or "對話及任務結束，祝您有美好的一天～")
    if node_name == "input_node" and state_update.get("task"):
        task = state_update.get("task", {}) or {}
        label = task.get("normalized_task") or task.get("original_user_request", "")
        return f"任務已確認：{label}" if label else ""
    return ""



def _progress_message_from_update(
    node_name: str,
    merged_state: Dict[str, Any],
    state_update: Dict[str, Any],
) -> str:
    status = str(state_update.get("current_status", "") or "")
    if not status or status in {"GREETING_SENT", "GREETING_SKIPPED", "AI_REPLY_SENT", "GOODBYE_SENT"}:
        return ""

    if node_name == "human_reply_node":
        return "已收到你的訊息。接下來會判斷這是一般聊天，還是需要機器人執行的抓取任務。"

    if node_name == "task_classification_node":
        intent = str(state_update.get("task_intent", "") or "")
        if intent == "general_chat":
            return "已判斷為一般聊天。接下來產生聊天回覆。"
        return "已判斷為機器人抓取任務。接下來確認任務文字，並由 LLM 從 objects.yaml 搜尋相關物件候選。"

    if node_name == "chat_memory_node":
        return "已更新聊天記憶。這一輪一般聊天已完成。"

    if node_name == "input_node":
        if status == "INPUT_RECEIVED":
            task = state_update.get("task", merged_state.get("task", {})) or {}
            task_label = task.get("normalized_task") or task.get("original_user_request", "這個任務")
            return f"已確認任務：{task_label}。接下來搜尋 world_position 裡的目標 instance。"
        return "尚未確認合法的任務目標。接下來會結束這輪任務。"

    if node_name == "find_node":
        if status == "TARGET_SELECTED_FROM_WORLD_POSITION":
            target = state_update.get("selected_instance", {}) or {}
            name = target.get("instance_key") or target.get("item_id") or "目標 instance"
            return f"已選定候選目標：{name}。接下來整理目標的 3D 資訊與導航候選位置。"
        return "沒有找到可用的目標 instance。接下來返回結束流程。"

    if node_name == "get_item_info_no_sam3d_node":
        if status == "ITEM_INFO_NO_SAM3D_READY":
            return "已取得目標 3D 資訊與導航候選位置。接下來開始導航到目標附近。"
        return "目標 3D 資訊整理失敗。接下來返回 home 並結束任務。"

    if node_name == "nav_move_node":
        if status == "NAV_COMPLETED":
            return "已完成導航移動。接下來觀察目前環境，確認下一步動作。"
        return "導航移動沒有成功完成。接下來記錄結果並重新評估或返回 home。"

    if node_name == "observe_node":
        return "已取得目前環境觀察。接下來由模型推理下一步行動。"

    if node_name == "reason_node":
        decision = state_update.get("decision", merged_state.get("decision", {})) or {}
        module = str(decision.get("call_module", "") or "")
        if module == "DONE":
            return "已完成推理，任務達成結束條件。接下來返回 home。"
        if module in {"nav_agent", "major_nav_agent", "major_nav_node"}:
            return "已完成推理，決定調整導航位置。接下來準備下一個導航候選點。"
        if module in {"grasp_agent", "approach_agent", "car_approach_agent"}:
            return "已完成推理，決定進入抓取/靠近流程。接下來更新目標資訊。"
        return "已完成推理。接下來依照模型決策執行下一個節點。"

    if node_name in {"update_item_info_1_node", "update_item_info_2_node"}:
        world = state_update.get("world_position", merged_state.get("world_position", {})) or {}
        if world.get("target_changed"):
            return "已更新目標位置，且偵測到目標位置改變。接下來重新整理 3D 資訊。"
        if world.get("update_reason") == "target_missing":
            return "已更新目標位置，但目標消失。接下來返回 home。"
        if node_name == "update_item_info_1_node":
            return "已更新目標位置資訊。接下來準備下一個導航候選點。"
        return "已更新目標位置資訊。接下來執行抓取與靠近流程。"

    if node_name == "major_nav_node":
        if status == "MAJOR_NAV_CONTEXT_READY":
            navigation = state_update.get("navigation", merged_state.get("navigation", {})) or {}
            rank = navigation.get("current_goal_rank", "")
            return f"已準備第 {rank} 組導航候選點。接下來執行導航移動。"
        return "已沒有更多導航候選點。接下來返回 home 並結束任務。"

    if node_name == "car_grasp_node":
        return "已完成抓取規劃。接下來依照抓取結果控制車體與手臂靠近。"

    if node_name == "car_approach_node":
        if status == "APPROACH_COMPLETED":
            return "已完成靠近/執行動作。接下來返回 home。"
        return "靠近/執行動作沒有成功完成。接下來更新任務記憶並重新觀察環境。"

    if node_name == "update_memory_node":
        return "已更新任務記憶。接下來重新觀察環境，確認下一步。"

    if node_name == "nav_home_node":
        if status == "NAV_HOME_COMPLETED":
            return "已完成返回 home。接下來結束對話與任務。"
        return "返回 home 沒有成功完成。接下來仍會進入結束流程。"

    return ""


def _progress_message_from_interrupt(pending: Dict[str, Any]) -> str:
    interrupt_type = str(pending.get("type", "") or "")
    if interrupt_type == "detection_selection":
        return "已找到候選目標照片。接下來需要你在聊天卡片中選擇正確照片，或選擇 No valid target。"
    return "流程暫停等待你的輸入。接下來請在聊天卡片中完成選擇。"

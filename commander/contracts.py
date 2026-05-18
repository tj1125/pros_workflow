from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactRef(StrictModel):
    artifact_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    mime_type: str | None = None


class NodeExecution(StrictModel):
    node_name: str
    status: str
    success: bool = True
    message: str = ""
    latency_sec: float = 0.0
    error: str = ""
    route_to: str = ""


class TaskContext(StrictModel):
    task_id: str
    original_user_request: str
    normalized_task: str
    task_type: Literal["pick_and_place", "chat", "unknown"] = "pick_and_place"
    success_criteria: list[str] = Field(default_factory=list)
    done_policy: str = ""


class RequestedObject(StrictModel):
    id: str = ""
    label: str = ""


class SelectedInstance(StrictModel):
    item_id: str
    instance_id: int
    instance_key: str
    topic_key: str = ""
    center_world: list[float] = Field(default_factory=list)
    camsrc: list[str] = Field(default_factory=list)
    bboxes_by_camera: dict[str, list[float]] = Field(default_factory=dict)
    primary_camera: str = ""
    primary_bbox: list[float] = Field(default_factory=list)
    preview_ref: ArtifactRef | None = None


class WorldPositionSnapshot(StrictModel):
    snapshot_id: str = ""
    candidate_count: int = 0
    selected_instance_key: str = ""
    updated_at: float = 0.0
    target_changed: bool = False
    update_source_node: str = ""
    update_distance_m: float = 0.0
    update_reason: str = ""


class RoomCameraSnapshot(StrictModel):
    camera_name: str
    topic: str = ""
    image_key: str = ""
    rgb_ref: ArtifactRef | None = None
    bbox: list[float] = Field(default_factory=list)
    preview_path: str = ""
    preview_ref: ArtifactRef | None = None


class ItemInfoResult(StrictModel):
    center_world: list[float] = Field(default_factory=list)
    center_world_coordinate_frame: str = ""
    primary_camera_id: str = ""
    target_instance_key: str = ""
    target_topic_key: str = ""
    group_ranking: list[dict[str, Any]] = Field(default_factory=list)
    goal_pose_path: str = ""
    raw_result_ref: ArtifactRef | None = None
    a2a_task_id: str = ""


class NavGoal(StrictModel):
    x: float
    y: float
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float
    qw: float
    yaw: float | None = None
    goal_rank: int = 1
    goal_pose_index: int = 0
    goal_pose_source: str = ""
    face_target_x: float | None = None
    face_target_y: float | None = None
    orientation_group: Any | None = None
    grasp_confidence: float | None = None
    map_feasible: bool | None = None
    selection_mode: str = ""


class NavResult(StrictModel):
    goal: NavGoal | None = None
    arrived: bool = False
    plan_ready: bool = False
    attempt: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)
    message: str = ""


class NavigationState(StrictModel):
    current_goal_rank: int = 1
    current_goal_pose_index: int = 0
    goal_pose_db: dict[str, Any] = Field(default_factory=dict)
    nav_goal: NavGoal | None = None
    nav_goal_pose_source: str = ""
    nav_move_source: str = ""
    force_initialpose: bool = False
    result: NavResult | None = None


class Observation(StrictModel):
    description: str = ""
    image_key: str = ""
    image_ref: ArtifactRef | None = None


class DecisionRecord(StrictModel):
    reasoning: str = ""
    call_module: str = ""
    module_params: dict[str, Any] = Field(default_factory=dict)
    latency_sec: float = 0.0
    model: str = ""


class GraspResult(StrictModel):
    object_id: str = ""
    camera_name: str = "Camera_Car"
    success: bool = False
    grasp_confidence: float | None = None
    num_candidate_grasps: int | None = None
    num_valid_grasps: int | None = None
    best_grasp_pose_camera: dict[str, Any] = Field(default_factory=dict)
    valid_grasp_poses_ref: ArtifactRef | None = None
    object_reference_center_camera: list[float] = Field(default_factory=list)
    rgb_ref: ArtifactRef | None = None
    depth_ref: ArtifactRef | None = None
    raw_result_ref: ArtifactRef | None = None
    a2a_task_id: str = ""


class ApproachResult(StrictModel):
    success: bool = False
    status_code: str = ""
    phase: str = ""
    message: str = ""
    next_agent: str | None = None
    nav_result: dict[str, Any] = Field(default_factory=dict)
    arm_result: dict[str, Any] = Field(default_factory=dict)
    arm_base_alignment_result: dict[str, Any] = Field(default_factory=dict)
    selected_solution: dict[str, Any] = Field(default_factory=dict)
    raw_result_ref: ArtifactRef | None = None


def dump_model(value: BaseModel | None) -> dict[str, Any]:
    if value is None:
        return {}
    return value.model_dump(mode="json", exclude_none=True)


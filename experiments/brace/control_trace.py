"""BRACE schema v2: control trace and branch snapshot HDF5 helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
REQUIRED_GROUPS = (
    "meta",
    "policy_chunks",
    "control_trace",
    "branch_snapshots",
)


def snapshot_indices(total_steps: int, count: int) -> list[int]:
    """Interior quartile physics-step indices for branch snapshots."""
    if total_steps < 2:
        raise ValueError("need at least two physics steps for snapshots")
    if count < 1:
        raise ValueError("snapshot count must be positive")
    available = total_steps
    if available < count:
        raise ValueError(f"only {available} physics steps for {count} snapshots")
    values = np.linspace(0, total_steps - 1, count + 2, dtype=np.int64)[1:-1]
    indices = sorted(set(int(value) for value in values))
    if len(indices) != count:
        indices = [int(value) for value in np.linspace(0, total_steps - 1, count, dtype=np.int64)]
    if len(set(indices)) != count:
        raise ValueError(f"cannot choose {count} unique snapshots from {total_steps} steps")
    return indices


def chunk_boundary_snapshot_indices(control_steps: list[dict[str, Any]]) -> list[int]:
    """Buffer indices of the last control step of each policy chunk (except the final chunk).

    A snapshot placed at the last step of chunk k-1 branches at the start of
    chunk k with an empty replay window, so this yields one snapshot per
    policy-chunk boundary.
    """
    indices: list[int] = []
    for index in range(len(control_steps) - 1):
        current_chunk = int(control_steps[index]["policy_chunk_index"])
        next_chunk = int(control_steps[index + 1]["policy_chunk_index"])
        if next_chunk != current_chunk:
            indices.append(index)
    return indices


@dataclass
class BranchContext:
    snapshot_id: int
    snapshot_physics_step: int
    boundary_physics_step: int
    branch_chunk_index: int
    replay_steps: list[dict[str, Any]]
    take_action_cnt: int

    @property
    def runtime_state(self) -> dict[str, Any]:
        return {
            "physics_step": int(self.boundary_physics_step),
            "take_action_cnt": int(self.take_action_cnt),
            "policy_chunk_index": int(self.branch_chunk_index) - 1,
            "eval_success": False,
        }


def buffer_index_for_physics_step(control_steps: list[dict[str, Any]], physics_step: int) -> int:
    for index, step in enumerate(control_steps):
        if int(step["physics_step"]) == int(physics_step):
            return index
    raise ValueError(f"physics_step {physics_step} not found in control trace")


def policy_chunk_index_at_physics_step(trace: dict[str, Any], physics_step: int) -> int:
    for step in trace["control_steps"]:
        if int(step["physics_step"]) == int(physics_step):
            return int(step["policy_chunk_index"])
    raise ValueError(f"physics_step {physics_step} not found in control trace")


def take_action_cnt_before_chunk(trace: dict[str, Any], branch_chunk_index: int) -> int:
    total = 0
    for chunk in trace["policy_chunks"]:
        chunk_index = int(chunk["chunk_index"])
        if chunk_index < branch_chunk_index:
            total += len(policy_chunk_actions(chunk["action"]))
        elif chunk_index == branch_chunk_index:
            break
    return total


def trace_has_chunk_index(trace: dict[str, Any], chunk_index: int) -> bool:
    return any(int(chunk["chunk_index"]) == int(chunk_index) for chunk in trace["policy_chunks"])


def load_policy_chunk_indices(hdf5_path: Path) -> set[int]:
    """Read only /policy_chunks/chunk_index (fast path for dataset export)."""
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        if root.attrs.get("brace_schema_version", 0) != SCHEMA_VERSION:
            raise ValueError(f"missing or unsupported brace_schema_version in {hdf5_path}")
        if "policy_chunks" not in root or "chunk_index" not in root["policy_chunks"]:
            raise ValueError(f"missing /policy_chunks/chunk_index in {hdf5_path}")
        return {int(value) for value in root["policy_chunks"]["chunk_index"][()]}


def build_branch_context(trace: dict[str, Any], snapshot: dict[str, Any]) -> BranchContext:
    control_steps = trace["control_steps"]
    if not control_steps:
        raise ValueError("control trace is empty")

    snapshot_physics_step = int(snapshot["physics_step"])
    start_idx = buffer_index_for_physics_step(control_steps, snapshot_physics_step)
    current_chunk = int(control_steps[start_idx]["policy_chunk_index"])

    end_of_chunk_idx = start_idx
    while (
        end_of_chunk_idx + 1 < len(control_steps)
        and int(control_steps[end_of_chunk_idx + 1]["policy_chunk_index"]) == current_chunk
    ):
        end_of_chunk_idx += 1

    branch_boundary_idx = end_of_chunk_idx + 1
    if branch_boundary_idx >= len(control_steps):
        raise ValueError(f"snapshot {snapshot.get('snapshot_id')} is too late for chunk-boundary branch")

    branch_chunk_index = int(control_steps[branch_boundary_idx]["policy_chunk_index"])
    if branch_chunk_index <= current_chunk:
        raise ValueError(
            f"snapshot {snapshot.get('snapshot_id')} did not advance to a later chunk "
            f"(current={current_chunk}, branch={branch_chunk_index})"
        )

    replay_steps = list(control_steps[start_idx + 1 : branch_boundary_idx])
    boundary_physics_step = int(control_steps[branch_boundary_idx]["physics_step"])
    take_action_cnt = take_action_cnt_before_chunk(trace, branch_chunk_index)

    return BranchContext(
        snapshot_id=int(snapshot["snapshot_id"]),
        snapshot_physics_step=snapshot_physics_step,
        boundary_physics_step=boundary_physics_step,
        branch_chunk_index=branch_chunk_index,
        replay_steps=replay_steps,
        take_action_cnt=take_action_cnt,
    )


def _resolve_entity(actor: Any) -> Any:
    """Return the underlying SAPIEN entity for Actor wrappers or raw entities."""
    wrapped = getattr(actor, "actor", None)
    if wrapped is not None and hasattr(wrapped, "get_components"):
        return wrapped
    return actor


def _rigid_dynamic_component(actor: Any):
    import sapien

    entity = _resolve_entity(actor)
    if hasattr(entity, "find_component_by_type"):
        component = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if component is not None:
            return component
    for component in entity.get_components():
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            return component
    return None


def _actor_velocity(actor: Any) -> tuple[np.ndarray, np.ndarray]:
    component = _rigid_dynamic_component(actor)
    if component is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    linear = getattr(component, "linear_velocity", None)
    angular = getattr(component, "angular_velocity", None)
    if linear is None and hasattr(component, "get_linear_velocity"):
        linear = component.get_linear_velocity()
    if angular is None and hasattr(component, "get_angular_velocity"):
        angular = component.get_angular_velocity()
    return (
        np.asarray(linear if linear is not None else np.zeros(3), dtype=np.float64),
        np.asarray(angular if angular is not None else np.zeros(3), dtype=np.float64),
    )


def actor_pose_vector(actor: Any) -> np.ndarray:
    pose = actor.get_pose()
    return np.concatenate((np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)))


def set_actor_pose_velocity(actor: Any, pose: np.ndarray, linear_velocity: np.ndarray, angular_velocity: np.ndarray) -> None:
    import sapien

    entity = _resolve_entity(actor)
    entity.set_pose(sapien.Pose(pose[:3], pose[3:7]))
    component = _rigid_dynamic_component(actor)
    if component is None:
        return
    if hasattr(component, "set_linear_velocity"):
        component.set_linear_velocity(linear_velocity)
        component.set_angular_velocity(angular_velocity)
    else:
        component.linear_velocity = linear_velocity
        component.angular_velocity = angular_velocity


def robot_state_dict(env: Any) -> dict[str, np.ndarray]:
    left_qpos = np.asarray(env.robot.left_entity.get_qpos(), dtype=np.float64)
    left_qvel = np.asarray(env.robot.left_entity.get_qvel(), dtype=np.float64)
    right_qpos = np.asarray(env.robot.right_entity.get_qpos(), dtype=np.float64)
    right_qvel = np.asarray(env.robot.right_entity.get_qvel(), dtype=np.float64)
    joints = np.asarray(
        env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
        dtype=np.float64,
    )
    end_effectors = {
        "left_endpose": np.asarray(env.get_arm_pose("left"), dtype=np.float64),
        "right_endpose": np.asarray(env.get_arm_pose("right"), dtype=np.float64),
    }
    objects: dict[str, np.ndarray] = {}
    for name, actor in env.get_dynamic_actors().items():
        pose = actor_pose_vector(actor)
        linear_velocity, angular_velocity = _actor_velocity(actor)
        objects[name] = {
            "pose": pose,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
    return {
        "left_qpos": left_qpos,
        "left_qvel": left_qvel,
        "right_qpos": right_qpos,
        "right_qvel": right_qvel,
        "joints": joints,
        "left_endpose": end_effectors["left_endpose"],
        "right_endpose": end_effectors["right_endpose"],
        "dynamic_actors": objects,
    }


def restore_robot_state(env: Any, state: dict[str, Any], *, settle: bool = False) -> None:
    env.robot.left_entity.set_qpos(np.asarray(state["left_qpos"], dtype=np.float64))
    env.robot.left_entity.set_qvel(np.asarray(state["left_qvel"], dtype=np.float64))
    env.robot.right_entity.set_qpos(np.asarray(state["right_qpos"], dtype=np.float64))
    env.robot.right_entity.set_qvel(np.asarray(state["right_qvel"], dtype=np.float64))
    for name, payload in state["dynamic_actors"].items():
        actor = env.get_dynamic_actors()[name]
        set_actor_pose_velocity(
            actor,
            np.asarray(payload["pose"], dtype=np.float64),
            np.asarray(payload["linear_velocity"], dtype=np.float64),
            np.asarray(payload["angular_velocity"], dtype=np.float64),
        )
    env._update_render()
    if settle:
        env.scene.step()


def replay_control_step(env: Any, step: dict[str, Any]) -> None:
    """Replay one recorded physics step without TOPP replanning."""
    env.robot.set_arm_joints(step["left_arm_pos"], step["left_arm_vel"], "left")
    env.robot.set_arm_joints(step["right_arm_pos"], step["right_arm_vel"], "right")
    env.robot.set_gripper(step["left_gripper"], "left")
    env.robot.set_gripper(step["right_gripper"], "right")
    env.scene.step()
    env._update_render()


def _quantize_cam(cam: Any) -> np.ndarray:
    """Store encode_obs camera frames (float32 = uint8/255) losslessly as uint8."""
    return np.round(np.asarray(cam, dtype=np.float64) * 255.0).astype(np.uint8)


def _dequantize_cam(cam: Any) -> np.ndarray:
    return (np.asarray(cam, dtype=np.float32) / 255.0).astype(np.float32)


def append_brace_trace_to_hdf5(
    hdf5_path: Path,
    *,
    policy_chunks: list[dict[str, Any]],
    control_steps: list[dict[str, Any]],
    branch_snapshots: list[dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    import h5py

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "a") as root:
        for group_name in REQUIRED_GROUPS:
            if group_name in root:
                del root[group_name]

        root.attrs["brace_schema_version"] = SCHEMA_VERSION
        meta_group = root.create_group("meta")
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)):
                meta_group.attrs[key] = value
            else:
                meta_group.attrs[key] = json.dumps(value)

        chunks = root.create_group("policy_chunks")
        if policy_chunks:
            chunks.create_dataset("chunk_index", data=np.asarray([row["chunk_index"] for row in policy_chunks], dtype=np.int64))
            chunks.create_dataset("policy_step", data=np.asarray([row["policy_step"] for row in policy_chunks], dtype=np.int64))
            chunks.create_dataset("physics_step", data=np.asarray([row["physics_step"] for row in policy_chunks], dtype=np.int64))
            chunks.create_dataset("action", data=np.asarray([row["action"] for row in policy_chunks], dtype=np.float64))

        trace = root.create_group("control_trace")
        if control_steps:
            trace.create_dataset("physics_step", data=np.asarray([row["physics_step"] for row in control_steps], dtype=np.int64))
            trace.create_dataset("policy_chunk_index", data=np.asarray([row["policy_chunk_index"] for row in control_steps], dtype=np.int64))
            trace.create_dataset("left_arm_pos", data=np.asarray([row["left_arm_pos"] for row in control_steps], dtype=np.float64))
            trace.create_dataset("left_arm_vel", data=np.asarray([row["left_arm_vel"] for row in control_steps], dtype=np.float64))
            trace.create_dataset("right_arm_pos", data=np.asarray([row["right_arm_pos"] for row in control_steps], dtype=np.float64))
            trace.create_dataset("right_arm_vel", data=np.asarray([row["right_arm_vel"] for row in control_steps], dtype=np.float64))
            trace.create_dataset("left_gripper", data=np.asarray([row["left_gripper"] for row in control_steps], dtype=np.float64))
            trace.create_dataset("right_gripper", data=np.asarray([row["right_gripper"] for row in control_steps], dtype=np.float64))

            robot = trace.create_group("robot_state")
            robot.create_dataset("left_qpos", data=np.asarray([row["robot_state"]["left_qpos"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("left_qvel", data=np.asarray([row["robot_state"]["left_qvel"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("right_qpos", data=np.asarray([row["robot_state"]["right_qpos"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("right_qvel", data=np.asarray([row["robot_state"]["right_qvel"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("joints", data=np.asarray([row["robot_state"]["joints"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("left_endpose", data=np.asarray([row["robot_state"]["left_endpose"] for row in control_steps], dtype=np.float64))
            robot.create_dataset("right_endpose", data=np.asarray([row["robot_state"]["right_endpose"] for row in control_steps], dtype=np.float64))

            actor_names = sorted(control_steps[0]["robot_state"]["dynamic_actors"].keys())
            actors = trace.create_group("dynamic_actors")
            for actor_name in actor_names:
                group = actors.create_group(actor_name)
                group.create_dataset(
                    "pose",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["pose"] for row in control_steps],
                        dtype=np.float64,
                    ),
                )
                group.create_dataset(
                    "linear_velocity",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["linear_velocity"] for row in control_steps],
                        dtype=np.float64,
                    ),
                )
                group.create_dataset(
                    "angular_velocity",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["angular_velocity"] for row in control_steps],
                        dtype=np.float64,
                    ),
                )

        snaps = root.create_group("branch_snapshots")
        if branch_snapshots:
            snaps.create_dataset("snapshot_id", data=np.asarray([row["snapshot_id"] for row in branch_snapshots], dtype=np.int64))
            snaps.create_dataset("physics_step", data=np.asarray([row["physics_step"] for row in branch_snapshots], dtype=np.int64))
            snaps.create_dataset("control_trace_offset", data=np.asarray([row["control_trace_offset"] for row in branch_snapshots], dtype=np.int64))

            snap_robot = snaps.create_group("robot_state")
            snap_robot.create_dataset("left_qpos", data=np.asarray([row["robot_state"]["left_qpos"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("left_qvel", data=np.asarray([row["robot_state"]["left_qvel"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("right_qpos", data=np.asarray([row["robot_state"]["right_qpos"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("right_qvel", data=np.asarray([row["robot_state"]["right_qvel"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("joints", data=np.asarray([row["robot_state"]["joints"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("left_endpose", data=np.asarray([row["robot_state"]["left_endpose"] for row in branch_snapshots], dtype=np.float64))
            snap_robot.create_dataset("right_endpose", data=np.asarray([row["robot_state"]["right_endpose"] for row in branch_snapshots], dtype=np.float64))

            snap_actor_names = sorted(branch_snapshots[0]["robot_state"]["dynamic_actors"].keys())
            snap_actors = snaps.create_group("dynamic_actors")
            for actor_name in snap_actor_names:
                group = snap_actors.create_group(actor_name)
                group.create_dataset(
                    "pose",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["pose"] for row in branch_snapshots],
                        dtype=np.float64,
                    ),
                )
                group.create_dataset(
                    "linear_velocity",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["linear_velocity"] for row in branch_snapshots],
                        dtype=np.float64,
                    ),
                )
                group.create_dataset(
                    "angular_velocity",
                    data=np.asarray(
                        [row["robot_state"]["dynamic_actors"][actor_name]["angular_velocity"] for row in branch_snapshots],
                        dtype=np.float64,
                    ),
                )

            if branch_snapshots[0].get("observation_joint_vector") is not None:
                snaps.create_dataset(
                    "observation_joint_vector",
                    data=np.asarray([row["observation_joint_vector"] for row in branch_snapshots], dtype=np.float64),
                )

            if all(row.get("obs_history") is not None for row in branch_snapshots):
                obs_group = snaps.create_group("obs_history")
                for cam_key in ("head_cam", "left_cam", "right_cam"):
                    obs_group.create_dataset(
                        cam_key,
                        data=np.asarray(
                            [_quantize_cam(row["obs_history"][cam_key]) for row in branch_snapshots],
                            dtype=np.uint8,
                        ),
                        compression="gzip",
                        compression_opts=4,
                    )
                obs_group.create_dataset(
                    "agent_pos",
                    data=np.asarray(
                        [row["obs_history"]["agent_pos"] for row in branch_snapshots], dtype=np.float64
                    ),
                )


def load_brace_trace(hdf5_path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, "r") as root:
        if root.attrs.get("brace_schema_version", 0) != SCHEMA_VERSION:
            raise ValueError(f"missing or unsupported brace_schema_version in {hdf5_path}")
        for group_name in REQUIRED_GROUPS:
            if group_name not in root:
                raise ValueError(f"missing /{group_name} in {hdf5_path}")

        trace = root["control_trace"]
        actor_names = sorted(root["control_trace"]["dynamic_actors"].keys())
        control_steps = []
        step_count = int(trace["physics_step"].shape[0])
        for index in range(step_count):
            dynamic_actors = {}
            for actor_name in actor_names:
                actor_group = trace["dynamic_actors"][actor_name]
                dynamic_actors[actor_name] = {
                    "pose": np.asarray(actor_group["pose"][index], dtype=np.float64),
                    "linear_velocity": np.asarray(actor_group["linear_velocity"][index], dtype=np.float64),
                    "angular_velocity": np.asarray(actor_group["angular_velocity"][index], dtype=np.float64),
                }
            control_steps.append(
                {
                    "physics_step": int(trace["physics_step"][index]),
                    "policy_chunk_index": int(trace["policy_chunk_index"][index]),
                    "left_arm_pos": np.asarray(trace["left_arm_pos"][index], dtype=np.float64),
                    "left_arm_vel": np.asarray(trace["left_arm_vel"][index], dtype=np.float64),
                    "right_arm_pos": np.asarray(trace["right_arm_pos"][index], dtype=np.float64),
                    "right_arm_vel": np.asarray(trace["right_arm_vel"][index], dtype=np.float64),
                    "left_gripper": float(trace["left_gripper"][index]),
                    "right_gripper": float(trace["right_gripper"][index]),
                    "robot_state": {
                        "left_qpos": np.asarray(trace["robot_state"]["left_qpos"][index], dtype=np.float64),
                        "left_qvel": np.asarray(trace["robot_state"]["left_qvel"][index], dtype=np.float64),
                        "right_qpos": np.asarray(trace["robot_state"]["right_qpos"][index], dtype=np.float64),
                        "right_qvel": np.asarray(trace["robot_state"]["right_qvel"][index], dtype=np.float64),
                        "joints": np.asarray(trace["robot_state"]["joints"][index], dtype=np.float64),
                        "left_endpose": np.asarray(trace["robot_state"]["left_endpose"][index], dtype=np.float64),
                        "right_endpose": np.asarray(trace["robot_state"]["right_endpose"][index], dtype=np.float64),
                        "dynamic_actors": dynamic_actors,
                    },
                }
            )

        snaps = root["branch_snapshots"]
        branch_snapshots = []
        if "physics_step" in snaps and int(snaps["physics_step"].shape[0]) > 0:
            snap_actor_names = sorted(snaps["dynamic_actors"].keys())
            snap_count = int(snaps["physics_step"].shape[0])
            for index in range(snap_count):
                dynamic_actors = {}
                for actor_name in snap_actor_names:
                    actor_group = snaps["dynamic_actors"][actor_name]
                    dynamic_actors[actor_name] = {
                        "pose": np.asarray(actor_group["pose"][index], dtype=np.float64),
                        "linear_velocity": np.asarray(actor_group["linear_velocity"][index], dtype=np.float64),
                        "angular_velocity": np.asarray(actor_group["angular_velocity"][index], dtype=np.float64),
                    }
                branch_snapshots.append(
                    {
                        "snapshot_id": int(snaps["snapshot_id"][index]),
                        "physics_step": int(snaps["physics_step"][index]),
                        "control_trace_offset": int(snaps["control_trace_offset"][index]),
                        "robot_state": {
                            "left_qpos": np.asarray(snaps["robot_state"]["left_qpos"][index], dtype=np.float64),
                            "left_qvel": np.asarray(snaps["robot_state"]["left_qvel"][index], dtype=np.float64),
                            "right_qpos": np.asarray(snaps["robot_state"]["right_qpos"][index], dtype=np.float64),
                            "right_qvel": np.asarray(snaps["robot_state"]["right_qvel"][index], dtype=np.float64),
                            "joints": np.asarray(snaps["robot_state"]["joints"][index], dtype=np.float64),
                            "left_endpose": np.asarray(snaps["robot_state"]["left_endpose"][index], dtype=np.float64),
                            "right_endpose": np.asarray(snaps["robot_state"]["right_endpose"][index], dtype=np.float64),
                            "dynamic_actors": dynamic_actors,
                        },
                        "observation_joint_vector": (
                            np.asarray(snaps["observation_joint_vector"][index], dtype=np.float64)
                            if "observation_joint_vector" in snaps
                            else None
                        ),
                        "obs_history": (
                            {
                                "head_cam": _dequantize_cam(snaps["obs_history"]["head_cam"][index]),
                                "left_cam": _dequantize_cam(snaps["obs_history"]["left_cam"][index]),
                                "right_cam": _dequantize_cam(snaps["obs_history"]["right_cam"][index]),
                                "agent_pos": np.asarray(snaps["obs_history"]["agent_pos"][index], dtype=np.float64),
                            }
                            if "obs_history" in snaps
                            else None
                        ),
                    }
                )

        chunks = root["policy_chunks"]
        policy_chunks = []
        for index in range(int(chunks["chunk_index"].shape[0])):
            policy_chunks.append(
                {
                    "chunk_index": int(chunks["chunk_index"][index]),
                    "policy_step": int(chunks["policy_step"][index]),
                    "physics_step": int(chunks["physics_step"][index]),
                    "action": policy_chunk_actions(np.asarray(chunks["action"][index], dtype=np.float64)),
                }
            )

        meta = {key: root["meta"].attrs[key] for key in root["meta"].attrs.keys()}
        return {
            "meta": meta,
            "policy_chunks": policy_chunks,
            "control_steps": control_steps,
            "branch_snapshots": branch_snapshots,
        }


def policy_chunk_actions(chunk_action: np.ndarray) -> np.ndarray:
    action_array = np.asarray(chunk_action, dtype=np.float64)
    if action_array.ndim == 1:
        return action_array[None, :]
    return action_array


def capture_model_obs_history(model: Any) -> dict[str, np.ndarray] | None:
    """Capture the observation frames the policy just consumed in get_action.

    Reads the DP runner's obs deque right after get_action: the last
    n_obs_steps entries (fewer at episode start, when the runner repeat-fills
    the earliest frame) are stacked oldest-to-newest into one array per key.
    """
    runner = getattr(model, "runner", None)
    if runner is None or not getattr(runner, "obs", None):
        return None
    n_obs_steps = int(getattr(runner, "n_obs_steps", 3))
    frames = list(runner.obs)[-n_obs_steps:]
    # Repeat-fill the earliest frame to n_obs_steps, matching the runner's
    # stack_last_n_obs padding: the stored history is exactly what the policy saw.
    while len(frames) < n_obs_steps:
        frames.insert(0, frames[0])
    return {
        "head_cam": np.stack([np.asarray(frame["head_cam"], dtype=np.float32) for frame in frames]),
        "left_cam": np.stack([np.asarray(frame["left_cam"], dtype=np.float32) for frame in frames]),
        "right_cam": np.stack([np.asarray(frame["right_cam"], dtype=np.float32) for frame in frames]),
        "agent_pos": np.stack([np.asarray(frame["agent_pos"], dtype=np.float64) for frame in frames]),
    }


def obs_history_frames(obs_history: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Split a stored per-snapshot obs history into per-frame encode_obs-style dicts.

    obs_history holds stacked arrays of shape [n_frames, ...] per key; the
    frames are ordered oldest to newest, the last frame being the observation
    at the chunk boundary itself.
    """
    n_frames = int(np.asarray(obs_history["agent_pos"]).shape[0])
    frames = []
    for frame_index in range(n_frames):
        frames.append(
            {
                "head_cam": np.asarray(obs_history["head_cam"][frame_index], dtype=np.float32),
                "left_cam": np.asarray(obs_history["left_cam"][frame_index], dtype=np.float32),
                "right_cam": np.asarray(obs_history["right_cam"][frame_index], dtype=np.float32),
                "agent_pos": np.asarray(obs_history["agent_pos"][frame_index], dtype=np.float64),
            }
        )
    return frames


def restore_model_obs_history(model: Any, obs_history: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
    """Restore the policy's observation deque from a stored snapshot history.

    Feeds all frames except the newest into the model via update_obs and
    returns the newest frame, which the caller passes to get_action (which
    appends it) — exactly matching collection-time deque semantics. Returns
    None (after a bare reset) when no history is stored, in which case the
    caller falls back to the live boundary observation.
    """
    model.reset_obs()
    if obs_history is None:
        return None
    frames = obs_history_frames(obs_history)
    if not frames:
        return None
    for frame in frames[:-1]:
        model.update_obs(frame)
    return frames[-1]


def validate_schema_v2(hdf5_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = load_brace_trace(hdf5_path)
    except Exception as exc:
        errors.append(f"{hdf5_path}: {type(exc).__name__}: {exc}")
        return errors

    meta = payload.get("meta", {})
    expected_steps = meta.get("n_action_steps")
    for chunk in payload.get("policy_chunks", []):
        action = np.asarray(chunk["action"], dtype=np.float64)
        if action.ndim == 1:
            if expected_steps not in (None, 1):
                errors.append(
                    f"{hdf5_path}: legacy single-step policy chunk with n_action_steps={expected_steps}"
                )
            continue
        if action.ndim != 2:
            errors.append(f"{hdf5_path}: policy chunk action has unsupported shape {action.shape}")
            continue
        if expected_steps is not None and int(action.shape[0]) != int(expected_steps):
            errors.append(
                f"{hdf5_path}: policy chunk action steps {action.shape[0]} != meta n_action_steps {expected_steps}"
            )
    return errors

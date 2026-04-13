from typing import Callable, List, Optional, Dict, Any

try:
    import gymnasium as gym
except Exception:
    import gym

from .arm_env import make_single_env


def make_vector_envs(
    num_envs: int = 3,
    use_subproc: bool = False,
    env_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create multiple environments. Uses gym.vector.SyncVectorEnv by default.
    If `use_subproc` is True and stable-baselines3 is available, SubprocVecEnv
    can be used; otherwise falls back to SyncVectorEnv.
    """
    env_kwargs = env_kwargs or {}

    def make_thunk() -> Callable[[], gym.Env]:
        return lambda: make_single_env(**env_kwargs)

    env_fns: List[Callable[[], gym.Env]] = [make_thunk() for _ in range(num_envs)]

    if use_subproc:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv

            return SubprocVecEnv(env_fns)
        except Exception:
            pass

    return gym.vector.SyncVectorEnv(env_fns)

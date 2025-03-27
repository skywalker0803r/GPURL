"""
env.py
"""

import gymnasium as gym
import numpy as np
import subprocess
import time

from gymnasium import spaces
from typing import Dict, Any, Tuple

from monitor import GPUInfoMonitor
from reward import compute_reward

class GPUEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, gpu_id=0, step_time=2.0):
        super().__init__()
        self.gpu_id = gpu_id
        self.step_time = step_time

        # Action space: 相對值 -50~+50
        self.action_space = spaces.Box(
            low=np.array([-50.0]),
            high=np.array([50.0]),
            shape=(1,),
            dtype=np.float32
        )

        # 觀測空間 (7)
        low_obs = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs= np.array([120.0, 100.0, 10.0, 300.0, 300.0, 1.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs,
            high=high_obs,
            shape=(7,),
            dtype=np.float32
        )

        self.monitor = GPUInfoMonitor(gpu_id=self.gpu_id)

        self.current_temp = 0.0
        self.current_slope_3s = 0.0
        self.current_power_limit = 260.0
        self.current_power_draw = 0.0
        self.current_eta = 0.0
        self.current_fan = 0.0
        self.current_utilization = 0.0

        self.pl_old = 260.0
        self.terminated_flag = False

        self.reset()

    def set_power_limit(self, pl_value: float):
        clamped_val = float(np.clip(pl_value, 100.0, 275.0))
        cmd = [
            "nvidia-smi",
            "-i", str(self.gpu_id),
            f"--power-limit={int(clamped_val)}"
        ]
        try:
            subprocess.check_call(cmd, shell=False)
        except subprocess.CalledProcessError as e:
            print("[WARN] Setting PL failed?", e)

    def set_fan_speed(self, fan_speed: float):
        """
        Set GPU fan speed using nvidia-settings
        Args:
            fan_speed: Fan speed value between 0-100
        """
        clamped_val = float(np.clip(fan_speed, 0.0, 100.0))
        cmd = [
            "nvidia-settings",
            "--assign",
            f"[fan:0]/GPUTargetFanSpeed={int(clamped_val)}"
        ]
        try:
            subprocess.check_call(cmd, shell=False)
        except subprocess.CalledProcessError as e:
            print("[WARN] Setting fan speed failed?", e)

    def stress_gpu(self):
        """
        Stress test the GPU using CUDA matrix operations
        """
        try:
            import torch
            # 創建大矩陣進行運算
            size = 8192
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            # 進行多次矩陣乘法運算
            for _ in range(100):  # 執行100次運算
                c = torch.matmul(a, b)
                torch.cuda.synchronize()  # 確保運算完成
        except Exception as e:
            print("[WARN] GPU stress test failed?", e)

    def get_temp(self) -> float:
        """
        Get current GPU temperature
        Returns:
            float: Current GPU temperature in Celsius
        """
        self.monitor.update_info()
        obs_dict = self.monitor.get_observation()
        self.current_temp = obs_dict["temp"]
        return self.current_temp

    def step(self, action: np.ndarray):
        delta = float(action[0])
        pl_new = self.pl_old + delta
        pl_new = float(np.clip(pl_new, 100.0, 275.0))

        self.set_power_limit(pl_new)
        time.sleep(self.step_time)

        self.monitor.update_info()
        obs_dict = self.monitor.get_observation()

        temp = obs_dict["temp"]
        terminated = False
        truncated = False

        reward = compute_reward(
            temp=temp,
            pl_old=self.pl_old,
            pl_new=pl_new
        )

        self.pl_old = pl_new
        self.terminated_flag = terminated

        self.current_temp        = obs_dict["temp"]
        self.current_slope_3s    = obs_dict["slope_3s"]
        self.current_power_limit = obs_dict["power_limit"]
        self.current_power_draw  = obs_dict["actual_power_draw"]
        self.current_eta         = obs_dict["eta"]
        self.current_fan         = obs_dict["fan"]
        self.current_utilization = obs_dict["gpu_util"]

        obs_arr = np.array([
            self.current_temp,
            self.current_utilization,
            self.current_slope_3s,
            self.current_power_limit,
            self.current_power_draw,
            self.current_eta,
            self.current_fan
        ], dtype=np.float32)

        return obs_arr, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.terminated_flag = False
        self.monitor.temp_history.clear()

        self.pl_old = 260.0
        self.set_power_limit(self.pl_old)
        time.sleep(1.0)

        self.monitor.update_info()
        obs_dict = self.monitor.get_observation()

        self.current_temp        = obs_dict["temp"]
        self.current_slope_3s    = obs_dict["slope_3s"]
        self.current_power_limit = obs_dict["power_limit"]
        self.current_power_draw  = obs_dict["actual_power_draw"]
        self.current_eta         = obs_dict["eta"]
        self.current_fan         = obs_dict["fan"]
        self.current_utilization = obs_dict["gpu_util"]

        obs_arr = np.array([
            self.current_temp,
            self.current_utilization,
            self.current_slope_3s,
            self.current_power_limit,
            self.current_power_draw,
            self.current_eta,
            self.current_fan
        ], dtype=np.float32)

        return obs_arr, {}

    def render(self):
        pass

    def close(self):
        pass


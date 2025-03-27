"""
custom_callback.py
記錄到 TensorBoard + 印在終端機 + 寫到 CSV
"""

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class GPUControlCallback(BaseCallback):
    def __init__(self, verbose=1, dump_interval=100, csv_path="training_log.csv"):
        super().__init__(verbose)
        self.dump_interval = dump_interval
        self.csv_path = csv_path

        self.header_written = False

        self.temp_log = []
        self.slope_3s_log = []
        self.pl_log = []
        self.power_draw_log = []
        self.eta_log = []
        self.fan_log = []
        self.util_log = []
        self.reward_log = []

        self.episode_reward = 0.0

    def _init_callback(self) -> None:
        """
        在訓練開始前呼叫，可用來初始化 CSV。
        """
        

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", 
                "temp", 
                "power_limit",
                "reward",
                "slope_3s",
                "power_draw",
                "eta",
                "fan",
                "util"
            ])
        self.header_written = True

    def _on_step(self) -> bool:
        env_ = self.training_env.envs[0].unwrapped

        temp  = float(env_.current_temp)
        slope = float(env_.current_slope_3s)
        pl    = float(env_.current_power_limit)
        pdraw = float(env_.current_power_draw)
        eta   = float(env_.current_eta)
        fan   = float(env_.current_fan)
        util  = float(env_.current_utilization)

        rewards = self.locals.get('rewards', None)
        if rewards is not None and len(rewards) > 0:
            step_reward = float(rewards[0])
        else:
            step_reward = 0.0

        self.temp_log.append(temp)
        self.slope_3s_log.append(slope)
        self.pl_log.append(pl)
        self.power_draw_log.append(pdraw)
        self.eta_log.append(eta)
        self.fan_log.append(fan)
        self.util_log.append(util)
        self.reward_log.append(step_reward)

        self.episode_reward += step_reward

        # 在終端機印出
        if self.verbose > 0:
            print(
                f"[Step {self.num_timesteps}] "
                f"Temp={temp:.1f}, Slope={slope:.2f}, PL={pl:.1f}, PDraw={pdraw:.1f}, "
                f"ETA={eta:.2f}, Fan={fan:.1f}, Util={util:.1f}, "
                f"Reward={step_reward:.3f}"
            )

        # 寫入 TensorBoard
        self.logger.record("gpu/temp", temp)
        self.logger.record("gpu/slope_3s", slope)
        self.logger.record("gpu/power_limit", pl)
        self.logger.record("gpu/power_draw", pdraw)
        self.logger.record("gpu/eta", eta)
        self.logger.record("gpu/fan", fan)
        self.logger.record("gpu/utilization", util)
        self.logger.record("gpu/reward", step_reward)

        # 若回合結束(假如有設 done=True)
        dones = self.locals.get('dones', None)
        if dones is not None and len(dones) > 0 and dones[0]:
            print(f"Episode done. Total episode reward: {self.episode_reward:.3f}")
            self.episode_reward = 0.0

        # 將本步資訊寫入 CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                temp,
                pl,
                step_reward,
                slope,
                pdraw,
                eta,
                fan,
                util
            ])

        # 每隔 dump_interval 步強制將 TensorBoard logger dump
        if (self.num_timesteps % self.dump_interval) == 0:
            self.logger.dump(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        pass


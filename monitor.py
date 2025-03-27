"""
monitor.py
讀取 GPU 資訊 (nvidia-smi)，供環境使用。
"""
import subprocess
import time
from collections import deque

class GPUInfoMonitor:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.history_secs = 10.0
        self.temp_history = deque()

        self.current_temp = 0.0
        self.current_gpu_util = 0.0
        self.current_power_draw = 0.0
        self.current_power_limit = 150.0
        self.current_fan_speed = 0.0

    def _parse_nvidia_smi(self):
        command = [
            "nvidia-smi",
            f"--query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit,fan.speed",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_id}"
        ]
        try:
            output = subprocess.check_output(command, universal_newlines=True)
        except Exception as e:
            print(f"[WARN] Could not parse nvidia-smi: {e}")
            return (0.0, 0.0, 0.0, 150.0, 0.0)

        line = output.strip()
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            print(f"[WARN] nvidia-smi 回傳格式不足: {line}")
            return (0.0, 0.0, 0.0, 150.0, 0.0)

        try:
            temp = float(parts[0])
            gpu_util = float(parts[1])
            power_draw = float(parts[2])
            power_limit = float(parts[3])
            fan_speed = float(parts[4])
        except ValueError as ve:
            print(f"[WARN] 解析 nvidia-smi 輸出失敗: {ve}")
            return (0.0, 0.0, 0.0, 150.0, 0.0)

        return temp, gpu_util, power_draw, power_limit, fan_speed

    def update_info(self):
        now = time.time()
        temp, gpu_util, power_draw, power_limit, fan_speed = self._parse_nvidia_smi()

        self.temp_history.append((now, temp))
        while len(self.temp_history) > 0:
            oldest_time = self.temp_history[0][0]
            if now - oldest_time > self.history_secs:
                self.temp_history.popleft()
            else:
                break

        self.current_temp = temp
        self.current_gpu_util = gpu_util
        self.current_power_draw = power_draw
        self.current_power_limit = power_limit
        self.current_fan_speed = fan_speed

    def get_slope_3s(self):
        now = time.time()
        target_time = now - 3.0
        if len(self.temp_history) < 2:
            return 0.0

        reversed_history = reversed(self.temp_history)
        candidate = None
        for tstamp, ttemp in reversed_history:
            if tstamp <= target_time:
                candidate = (tstamp, ttemp)
                break
        if candidate is None:
            return 0.0

        older_time, older_temp = candidate
        current_temp = self.current_temp
        dt = now - older_time
        if dt < 1e-6:
            return 0.0
        slope = (current_temp - older_temp) / dt
        return slope

    def get_observation(self):
        slope_3s = self.get_slope_3s()
        if self.current_power_limit > 1e-6:
            eta = self.current_power_draw / self.current_power_limit
        else:
            eta = 0.0
        if eta > 1.0:
            eta = 1.0

        obs = {
            "temp": self.current_temp,
            "gpu_util": self.current_gpu_util,
            "slope_3s": slope_3s,
            "power_limit": self.current_power_limit,
            "actual_power_draw": self.current_power_draw,
            "eta": eta,
            "fan": self.current_fan_speed
        }
        return obs


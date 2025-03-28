from stable_baselines3 import PPO
from env import GPUEnv
import time
import os
from stable_baselines3.common.logger import configure

def test_model(model_path="modelA.zip", test_steps=100, log_name="test_run"):
    # 初始化環境
    env = GPUEnv(gpu_id=0, step_time=2.0)
    obs,info = env.reset()

    # 設定風扇轉速
    fan_speed_setting = 100
    env.set_fan_speed(fan_speed_setting)
    print(f"已設定風扇速度為 {fan_speed_setting}%")

    # 載入訓練好的模型
    model = PPO.load(model_path)

    # 設定 TensorBoard logger（會輸出到 tb_logs/test_run/）
    log_dir = f"./tb_logs/{log_name}"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["tensorboard"])
    model.set_logger(new_logger)

    # 開始測試
    for step in range(test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # TensorBoard scalar logs
        model.logger.record("test/temperature", env.current_temp)
        model.logger.record("test/power_limit", env.current_power_limit)
        model.logger.record("test/reward", reward)
        model.logger.record("test/step", step)
        model.logger.dump(step)

        # Console log
        print(f"[Step {step}] Temp: {env.current_temp}°C | PL: {env.current_power_limit}W | Reward: {reward:.4f}")

        if done:
            print("Episode done. Resetting environment.")
            obs = env.reset()

        time.sleep(2.0)

    print("測試完成")

if __name__ == "__main__":
    test_model()

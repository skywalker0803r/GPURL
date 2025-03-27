import time
import numpy as np
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from env import GPUEnv

def main():
    # 0) TensorBoard 日誌路徑 (區別於訓練資料)
    inference_logdir = "./tb_inference_logs/"
    writer = SummaryWriter(log_dir=inference_logdir)

    # 1) 建立環境
    env = GPUEnv(gpu_id=0, step_time=2.0)

    # 2) 載入已訓練的 model
    model_path = "modelA_B_C_D.zip"
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}.")

    # 3) 指定「我要在推理階段跑多少個step」 (無回合概念亦可)
    max_inference_steps = 10000

    obs, info = env.reset()
    global_step = 0
    ep_reward = 0.0
    episode_count = 0

    for step in range(max_inference_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_reward += reward
        global_step += 1

        # 紀錄到TensorBoard
        writer.add_scalar("inference/reward", reward, global_step)

        current_temp = getattr(env, "current_temp", np.nan)
        writer.add_scalar("inference/temp", current_temp, global_step)

        current_pl = getattr(env, "current_power_limit", np.nan)
        writer.add_scalar("inference/power_limit", current_pl, global_step)

        # 如果你還想記錄動作
        writer.add_scalar("inference/action", action[0], global_step)

        # 若你的環境很少會 done，不過還是檢查
        if terminated or truncated:
            print(f"Episode {episode_count} done, total reward={ep_reward:.3f}")
            obs, info = env.reset()
            ep_reward = 0.0
            episode_count += 1

        # 你也可以睡眠1秒或更短
        # time.sleep(1.0)

    # 4) 跑完1000步後，關閉
    env.close()
    writer.close()
    print("Inference ended after 1000 steps, logs in", inference_logdir)

if __name__ == "__main__":
    main()


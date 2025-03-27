"""
train_B.py

在「場景 B」下接續「場景 A」的模型 (modelA.zip)，繼續訓練。
需確保:
  - env.py, custom_callback.py, monitor.py, reward.py 與 train.py 同一層
  - modelA.zip 為上次訓練完成後保存的模型
"""

from stable_baselines3 import PPO
from env import GPUEnv
from stable_baselines3.common.callbacks import CallbackList
from custom_callback import GPUControlCallback

def main():
    # =============================================
    # 1. 準備環境
    # =============================================
    # 若「場景 B」需要在程式中動態切換風扇或散熱條件，可在這裡做:
    # (若是實體硬體, 可能要手動/外部指令把 fan=70% -> fan=30%)
    env = GPUEnv(gpu_id=0, step_time=2.0)

    # =============================================
    # 2. 載入之前的模型 (modelA.zip)
    # =============================================
    # 指定新的環境給這個模型
    model = PPO.load("modelA.zip", env=env, tensorboard_log="./tb_logs_fast/")
    print("載入 modelA.zip 成功，準備繼續在『場景 B』訓練...")

    # =============================================
    # 3. 建立自訂 Callback (Optional)
    # =============================================
    callback = GPUControlCallback(verbose=1, dump_interval=100, csv_path="training_log.csv")

    # =============================================
    # 4. 繼續學習
    # =============================================
    # 使用 reset_num_timesteps=False，延續之前的 time steps
    # total_timesteps: 依你需求設定
    model.learn(total_timesteps=15000, 
                callback=callback,
                log_interval=1,
                reset_num_timesteps=False)

    # =============================================
    # 5. 存檔 (modelA_B.zip)
    # =============================================
    model.save("modelA_B.zip")
    print("場景 B 訓練完成，已保存為 modelA_B.zip")

if __name__ == "__main__":
    main()

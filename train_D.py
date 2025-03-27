"""
train_D.py

在「場景 D」下接續「場景 C」的模型 (modelA_B_C.zip)，繼續訓練。
需確保:
  - env.py, custom_callback.py, monitor.py, reward.py 與 train.py 同一層
  - modelA_B_C.zip 為上次訓練完成後保存的模型
"""

from stable_baselines3 import PPO
from env import GPUEnv
from stable_baselines3.common.callbacks import CallbackList
from custom_callback import GPUControlCallback

def main():
    # =============================================
    # 1. 準備環境
    # =============================================
    # 若「場景 D」需要在程式中動態切換風扇或散熱條件，可在這裡做:
    
    env = GPUEnv(gpu_id=0, step_time=2.0)

    # =============================================
    # 2. 載入之前的模型 (modelA_B_C.zip)
    # =============================================
    # 指定新的環境給這個模型
    model = PPO.load("modelA_B_C.zip", env=env)
    print("載入 modelA_B_C.zip 成功，準備繼續在『場景 D』訓練...")

    # =============================================
    # 3. 建立自訂 Callback (Optional)
    # =============================================
    callback = GPUControlCallback(verbose=1, dump_interval=100)

    # =============================================
    # 4. 繼續學習
    # =============================================
    # 使用 reset_num_timesteps=False，延續之前的 time steps
    # total_timesteps: 依你需求設定
    model.learn(total_timesteps=12000, 
                callback=callback, 
                reset_num_timesteps=False)

    # =============================================
    # 5. 存檔 (modelA_B_C_D.zip)
    # =============================================
    model.save("modelA_B_C_D.zip")
    print("場景 D 訓練完成，已保存為 modelA_B_C_D.zip")

if __name__ == "__main__":
    main()

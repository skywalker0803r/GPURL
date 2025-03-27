"""
train.py

 - 使用 PPO 訓練 GPUEnv
 - 動作: 相對調整PL
 - 獎勵: 
   1) T>=75 => 0
   2) abs(pl_new - pl_old)>=20 => 0
   3) 否則 => norm_pl*(temp_factor)
 - Callback: 會同時寫CSV + TensorBoard
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from env import GPUEnv
from custom_callback import GPUControlCallback

def main():
    env = GPUEnv(gpu_id=0, step_time=2.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(log_std_init=5.0),
    )

    cb = GPUControlCallback(
        run_name = 'policy_kwargs=dict(log_std_init=5.0)',
        verbose=1, 
        dump_interval=100,
        csv_path="training_log.csv"  # CSV 檔案名稱
    )

    model.learn(
        total_timesteps=15000, 
        callback=cb,
        log_interval=1
    )

    model.save("modelA.zip")

if __name__=="__main__":
    main()


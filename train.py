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
from custom_callback import GPUControlCallback,EarlyStopCallback

'''
場景 A: 模擬溫度由安全區到70度之前要學會緩慢提升 PL
在 fan=100%下,燒到65度(fan=100%,燒機時的穩態),此時開始訓練,訓練開始的起
始 PL 是250W,所以在這個時候要會逐漸往上拉高 Power limit。
訓練100000步。
'''

train_name = '25250327train_A'

def main():
    env = GPUEnv(gpu_id=0, step_time=2.0)
    # 設定風扇轉速
    fan_speed_setting = 100
    env.set_fan_speed(fan_speed_setting)
    print(f'已設置fan_speed:{fan_speed_setting}') 
    # 燒到65度
    target_temp = 55
    while env.get_temp() <= target_temp:
        env.stress_gpu()
        print(f'正在進行升溫目標:{target_temp},目前:{env.current_temp}')
    # 設定PL 250W
    power_limit_setting = 250
    env.set_power_limit(power_limit_setting)
    print(f'已設置power limit:{power_limit_setting}')
    
    # 開始訓練
    print('開始訓練')

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./tb_logs/{train_name}",
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(log_std_init=1.0),
    )

    tensorboard_cb = GPUControlCallback(
        verbose=1, 
        dump_interval=1,
        csv_path="training_log.csv"  # CSV 檔案名稱
    )

    # 组合回调：先执行 TensorBoard 记录，再检查步数
    combined_cb = CallbackList([tensorboard_cb, EarlyStopCallback(max_steps=10)])

    model.learn(
    total_timesteps=10000,  # 保留此参数（虽然实际由 EarlyStopCallback 控制）
    callback=combined_cb,
    log_interval=1
    )

    model.save("modelA.zip")
    print('訓練結束')

if __name__=="__main__":
    main()


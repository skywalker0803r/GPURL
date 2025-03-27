from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from env import GPUEnv
from custom_callback import GPUControlCallback,EarlyStopCallback
from torch.utils.tensorboard import SummaryWriter


train_name = 'train_reward_A'
writer = SummaryWriter(log_dir=f"./tb_logs/{train_name}")
import inspect

def reward_1(temp, pl_old, pl_new):
    if temp >= 80:
        return -100
    elif temp >= 75:
        return -5 * (temp - 75)
    else:
        return (pl_new - 100) / 175 - (temp / 75)**2

reward_source = inspect.getsource(reward_1)
writer.add_text("Reward Function", f"```python\n{reward_source}\n```", global_step=0)
writer.close()

def main():
    env = GPUEnv(gpu_id=0, step_time=2.0)
    env.set_reward(reward_1)
    # 設定風扇轉速
    fan_speed_setting = 100
    env.set_fan_speed(fan_speed_setting)
    print(f'已設置fan_speed:{fan_speed_setting}') 
    # 燒到65度
    target_temp = 65
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
    combined_cb = CallbackList([tensorboard_cb, EarlyStopCallback(max_steps=10000)])

    model.learn(
    total_timesteps=10000,  # 保留此参数（虽然实际由 EarlyStopCallback 控制）
    callback=combined_cb,
    log_interval=1
    )

    model.save("modelA.zip")
    print('訓練結束')

if __name__=="__main__":
    main()


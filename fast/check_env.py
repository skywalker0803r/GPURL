from stable_baselines3.common.env_checker import check_env
from env import GPUEnv  # 從 env.py 中匯入 GPUEnv

def main():
    # 建立環境實例，參數可依實際需求調整
    env = GPUEnv(gpu_id=0, step_time=2.0)

    # 使用 check_env 來檢查是否符合 SB3 標準
    check_env(env)

    print("✅ 環境檢查完成！")

if __name__ == "__main__":
    main()


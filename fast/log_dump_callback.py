from stable_baselines3.common.callbacks import BaseCallback

class LogDumpCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 即使不用做任何事，也要回傳 True，
        # 讓這個抽象方法完整實作
        return True

    def _on_event(self) -> bool:
        if self.verbose > 0:
            print(f"[LogDumpCallback] Dump logger at step {self.num_timesteps}")
        self.logger.dump(self.num_timesteps)
        return True


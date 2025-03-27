def compute_reward(
    temp: float,
    pl_old: float,
    pl_new: float
) -> float:
    """
    設計邏輯:
      1) 若一次動作 >=30W => 0 分 (大動作 => 0)
      2) 若 temp>=75°C:
         - 若 pl_new < pl_old => 正獎勵:
             獎勵 = scale * (pl_old - pl_new) * early_factor * extra_factor
             其中:
               (a) pl_old - pl_new: 降幅
               (b) early_factor = (1 - norm_pl_old)
                  => pl_old越低 =>值越大, 鼓勵"越早就不拉那麼高"
               (c) extra_factor => 依"降得越多再多乘上一個額外系數"
         - 否則 (pl_new >= pl_old) => 負分
      3) 若 temp<75°C => 與原先類似: reward = norm_pl × temp_factor
    """
    """
    # (A) 大動作檢查
    delta_pl = abs(pl_new - pl_old)
    if delta_pl >= 30.0:
        return 0.0
    """
    
    # (B) 溫度 >=75°C
    if temp >= 75.0:
        if pl_new < pl_old:
            # 下調 => 給正獎勵

            # 1) 降幅
            drop_amount = pl_old - pl_new

            # 2) 早控因子 => pl_old越低 => 值越大
            norm_pl_old = (pl_old - 100.0) / 175.0
            if norm_pl_old < 0: norm_pl_old = 0
            if norm_pl_old > 1: norm_pl_old = 1
            early_factor = 1.0 - norm_pl_old

            # 3) 額外獎勵因子 => "降得越多，再額外放大"
            #   這裡示範一種簡單做法: extra_factor = 1 + 0.05*(drop_amount)
            #   => 降1W額外+0.05, 降10W額外+0.5, ...
            #   可視自己需求再調
            extra_factor = 1.0 + 0.05 * drop_amount
            if extra_factor < 1.0:
                extra_factor = 1.0  # 確保不小於1

            # 4) scale => 自己調整
            scale = 0.5

            reward = scale * drop_amount * early_factor * extra_factor

        else:
            # 在高溫下還升PL or不動 => 負分
            reward = -0.5

        return reward

    # (C) 溫度 <75 => 沿用原本正獎勵

    # pl factor
    norm_pl = (pl_new - 100.0) / 175.0
    if norm_pl < 0:
        norm_pl = 0
    elif norm_pl > 1:
        norm_pl = 1

    # temp factor
    if temp <= 70.0:
        temp_factor = 1.0
    else:
        # 線性: 70->1, 75->0
        temp_factor = (75.0 - temp) / 5.0

    reward = norm_pl * temp_factor
    return reward


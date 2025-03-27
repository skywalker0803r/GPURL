def compute_reward(
    temp: float,
    pl_old: float,
    pl_new: float
) -> float:
    """
    獎勵設計:
      (A) temp >= 75°C => 鼓勵大幅下調 (越早下調越大獎勵)
          - if pl_new < pl_old:
              drop_amount = pl_old - pl_new
              early_factor = (1 - norm_pl_old) 但若<0.1 => 0.1
              extra_factor = 1 + 0.3 * drop_amount
              reward = scale * drop_amount * early_factor * extra_factor
            else:
              reward = -1.0

      (B) 70°C <= temp < 75°C => 沿用原先 "norm_pl * temp_factor"

      (C) temp < 70°C => 鼓勵大幅上調 (越早上調越大獎勵)
          - if pl_new > pl_old:
              up_amount = pl_new - pl_old
              early_factor = norm_pl_old  但若<0.1 => 0.1
              extra_factor = 1 + 0.3 * up_amount
              reward = scale * up_amount * early_factor * extra_factor
            else:
              reward = -1.0

    說明:
      1) 為避免在 pl_old=275 (高溫) 或 pl_old=100 (低溫) 時 early_factor=0,
         我們在計算完 early_factor 後，增加一個下限(0.1)，
         使得微幅調整(1W)也能獲得少許正獎勵，引導 agent 擺脫極端值卡住情況。
      2) extra_factor = 1 + 0.3 * drop_amount/up_amount，可視需求調整倍率。
      3) scale = 0.5 同樣可依實驗需求自行調大或調小。
    """

    def get_norm_pl(val: float) -> float:
        # 將 PL 線性映射到 [0, 1]
        norm = (val - 100.0) / 175.0
        if norm < 0.0:
            norm = 0.0
        elif norm > 1.0:
            norm = 1.0
        return norm

    # ---------------------------
    # A) 高溫: temp >= 75°C
    # ---------------------------
    if temp >= 75.0:
        if pl_new < pl_old:
            drop_amount = pl_old - pl_new

            norm_pl_old = get_norm_pl(pl_old)
            # pl_old 越高 => (1 - norm_pl_old) 越小 => 原本會歸 0
            early_factor = 1.0 - norm_pl_old
            # 設定一個下限，避免 early_factor=0
            if early_factor < 0.1:
                early_factor = 0.1

            extra_factor = 1.0 + 0.3 * drop_amount
            if extra_factor < 1.0:
                extra_factor = 1.0

            scale = 0.5
            reward = scale * drop_amount * early_factor * extra_factor
        else:
            # 在高溫下未降(或持平) -> 負獎勵
            reward = -1.0
        return reward

    # --------------------------------
    # B) 中溫: 70°C <= temp < 75°C
    # --------------------------------
    if temp >= 70.0:
        norm_pl_new = get_norm_pl(pl_new)
        # 70°C->factor=1, 75°C->factor=0
        temp_factor = (75.0 - temp) / 5.0
        reward = norm_pl_new * temp_factor
        return reward

    # ---------------------------
    # C) 低溫: temp < 70°C
    # ---------------------------
    if temp < 70.0:
        if pl_new > pl_old:
            up_amount = pl_new - pl_old

            norm_pl_old = get_norm_pl(pl_old)
            # pl_old 越高 => norm_pl_old 越大 => early_factor 越大
            # 但若已壓到 100W => norm_pl_old=0 => early_factor=0 => 給下限
            early_factor = norm_pl_old
            if early_factor < 0.1:
                early_factor = 0.1

            extra_factor = 1.0 + 0.3 * up_amount
            if extra_factor < 1.0:
                extra_factor = 1.0

            scale = 0.5
            reward = scale * up_amount * early_factor * extra_factor
        else:
            # 在低溫下卻未上調(或持平/下調) -> 負獎勵
            reward = -1.0
        return reward


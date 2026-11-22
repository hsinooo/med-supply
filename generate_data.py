import os
import numpy as np
import pandas as pd


def generate():
    """產生模擬的耗材資料（從 2024-01-01 到昨天），存到 data 資料夾"""
    np.random.seed(42)  # 固定亂數，每次產生的資料都一樣
    os.makedirs("data", exist_ok=True)

    # 1. 耗材基本資料
    items = pd.DataFrame([
        {"耗材": "口罩",     "單位": "個", "每日平均用量": 120, "季節影響": 0.4, "目前庫存": 3000, "安全庫存": 1000},
        {"耗材": "紗布",     "單位": "包", "每日平均用量": 40,  "季節影響": 0.1, "目前庫存": 900,  "安全庫存": 300},
        {"耗材": "針筒",     "單位": "支", "每日平均用量": 80,  "季節影響": 0.2, "目前庫存": 2000, "安全庫存": 600},
        {"耗材": "酒精棉片", "單位": "片", "每日平均用量": 150, "季節影響": 0.3, "目前庫存": 4000, "安全庫存": 1200},
        {"耗材": "手套",     "單位": "雙", "每日平均用量": 200, "季節影響": 0.2, "目前庫存": 5000, "安全庫存": 1500},
        {"耗材": "點滴輸液套", "單位": "組", "每日平均用量": 30, "季節影響": 0.3, "目前庫存": 700,  "安全庫存": 250},
    ])

    # 2. 產生每天的用量紀錄
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)                # 資料一律產生到「昨天」
    dates = pd.date_range("2024-01-01", end, freq="D")
    rows = []
    for _, it in items.iterrows():
        for d in dates:
            season = 1 + it["季節影響"] * np.cos(2 * np.pi * (d.day_of_year - 15) / 365)  # 一月最高、七月最低
            weekday = 0.7 if d.day_of_week >= 5 else 1.0                                 # 週末用量較少
            trend = 1 + 0.0002 * (d - dates[0]).days                                     # 用量逐年微幅成長
            mean = it["每日平均用量"] * season * weekday * trend
            qty = max(0, int(np.random.normal(mean, mean * 0.1)))                         # 加入隨機波動
            rows.append({"日期": d.date(), "耗材": it["耗材"], "用量": qty})

    usage = pd.DataFrame(rows)

    # 3. 存成 CSV（utf-8-sig 讓 Excel 打開中文不會亂碼）
    items[["耗材", "單位", "目前庫存", "安全庫存"]].to_csv("data/items.csv", index=False, encoding="utf-8-sig")
    usage.to_csv("data/usage.csv", index=False, encoding="utf-8-sig")
    if os.path.exists("data/transactions.csv"):
        os.remove("data/transactions.csv")                                               # 資料重來，操作紀錄也清空
    return len(items), len(usage), dates[0].date(), dates[-1].date()


if __name__ == "__main__":
    n_items, n_rows, start, end = generate()
    print("資料產生完成！")
    print(f"耗材種類：{n_items} 種")
    print(f"用量紀錄：{n_rows} 筆（{start} ~ {end}）")

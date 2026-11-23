import hashlib
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from prophet import Prophet

from generate_data import MARKER, TZ, generate, taiwan_today

st.set_page_config(page_title="醫療耗材管理系統", page_icon="🏥", layout="wide")

ITEMS_FILE = "data/items.csv"
USAGE_FILE = "data/usage.csv"
LOG_FILE = "data/transactions.csv"


# ---------- 讀取資料 ----------
def ensure_data():
    """每天（台灣時間）第一次開啟時重新產生模擬資料，讓資料永遠接到昨天，展示環境也每天重設"""
    today = str(taiwan_today().date())
    if os.path.exists(ITEMS_FILE) and os.path.exists(USAGE_FILE) and os.path.exists(MARKER):
        with open(MARKER) as f:
            if f.read().strip() == today:
                return
    generate()


def load_data():
    items = pd.read_csv(ITEMS_FILE)
    usage = pd.read_csv(USAGE_FILE, parse_dates=["日期"])
    return items, usage


LOG_COLUMNS = ["時間", "操作者", "角色", "耗材", "類型", "數量", "異動後庫存"]


def load_log():
    if os.path.exists(LOG_FILE):
        log = pd.read_csv(LOG_FILE)
        return log.reindex(columns=LOG_COLUMNS)           # 舊的紀錄沒有操作者欄位，自動補上空白
    return pd.DataFrame(columns=LOG_COLUMNS)


def write_log(item, kind, qty, new_stock):
    log = load_log()
    user = st.session_state.get("user", {})
    log.loc[len(log)] = [datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"), user.get("name", "未知"),
                         user.get("role", "未知"), item, kind, qty, new_stock]
    log.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")


ensure_data()
items, usage = load_data()


# ---------- 領用與進貨：更新資料 ----------
def update_stock(item, kind, qty):
    items = pd.read_csv(ITEMS_FILE)
    idx = items.index[items["耗材"] == item][0]
    now_stock = int(items.at[idx, "目前庫存"])

    if kind == "領用":
        if qty > now_stock:
            return False, f"庫存不足！{item} 目前只剩 {now_stock}"
        new_stock = now_stock - qty

        # 領用量也要寫進每日用量紀錄，之後預測才會用到
        usage = pd.read_csv(USAGE_FILE)
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        mask = (usage["日期"] == today) & (usage["耗材"] == item)
        if mask.any():
            usage.loc[mask, "用量"] += qty
        else:
            usage.loc[len(usage)] = [today, item, qty]
        usage.to_csv(USAGE_FILE, index=False, encoding="utf-8-sig")
    else:  # 進貨
        new_stock = now_stock + qty

    # 更新庫存
    items.at[idx, "目前庫存"] = new_stock
    items.to_csv(ITEMS_FILE, index=False, encoding="utf-8-sig")

    # 寫入操作紀錄（含操作者）
    write_log(item, kind, qty, new_stock)

    return True, f"已{kind} {item} {qty}，目前庫存 {new_stock}"


# ---------- 計算庫存狀態 ----------
def build_overview(items, usage):
    today = taiwan_today()
    recent = usage[(usage["日期"] < today) & (usage["日期"] >= today - pd.Timedelta(days=30))]   # 最近 30 天（不含還沒過完的今天）
    avg = recent.groupby("耗材")["用量"].mean().round(1)

    df = items.copy()
    df["近30天平均日用量"] = df["耗材"].map(avg)
    df["預計可用天數"] = (df["目前庫存"] / df["近30天平均日用量"]).round(1)

    def status(row):
        if row["目前庫存"] < row["安全庫存"] or row["預計可用天數"] < 7:
            return "🔴 需立即補貨"
        elif row["預計可用天數"] < 14:
            return "🟡 注意"
        return "🟢 正常"

    df["狀態"] = df.apply(status, axis=1)
    return df



# ---------- AI 需求預測（Prophet） ----------
@st.cache_data(show_spinner=False)
def run_forecast(history, days):
    """history：DataFrame（日期、用量）；days：要預測幾天"""
    df = history.rename(columns={"日期": "ds", "用量": "y"})

    # 1. 準確度測試：用最後 30 天當「考卷」，只拿之前的資料訓練
    train, test = df.iloc[:-30], df.iloc[-30:]
    m_test = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m_test.fit(train)
    pred = m_test.predict(test[["ds"]])
    mape = (abs(test["y"].values - pred["yhat"].values) / test["y"].values).mean() * 100

    # 2. 用全部資料訓練，預測未來
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast, round(mape, 1)


# ---------- 登入與權限 ----------
# 密碼不直接存明碼，只存「加鹽雜湊值」（PBKDF2-SHA256）
def hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


USERS = {
    "nurse1": {"name": "王怡婷", "role": "護理師", "salt": "n9Xq2", "hash": hash_password("nurse111", "n9Xq2")},
    "nurse2": {"name": "陳雅雯", "role": "護理師", "salt": "k3Vm8", "hash": hash_password("nurse222", "k3Vm8")},
    "nurse3": {"name": "林志豪", "role": "護理師", "salt": "r6Tz1", "hash": hash_password("nurse333", "r6Tz1")},
    "admin":  {"name": "李明哲", "role": "管理員", "salt": "a7Lp5", "hash": hash_password("admin123", "a7Lp5")},
}

# 各角色能看到的頁面
PAGES = {
    "護理師": ["庫存總覽", "領用與進貨", "需求預測"],
    "管理員": ["庫存總覽", "領用與進貨", "需求預測", "補貨提醒", "操作紀錄"],
}

MAX_FAIL = 3        # 最多連續輸錯幾次
LOCK_SECONDS = 30   # 鎖定幾秒


@st.cache_resource
def login_guard():
    """記錄每個帳號輸錯幾次、鎖到什麼時候。存在伺服器端，重新整理網頁也無法繞過"""
    return {}


def login_page():
    st.title("🏥 AI 智慧醫療耗材管理與需求預測系統")
    st.subheader("🔐 請先登入")

    with st.form("login"):
        username = st.text_input("帳號").strip().lower()      # 去掉前後空白、統一小寫
        password = st.text_input("密碼", type="password").strip()
        submitted = st.form_submit_button("登入", type="primary")

    if submitted:
        guard = login_guard().setdefault(username, {"fails": 0, "lock_until": 0})
        user = USERS.get(username)

        if time.time() < guard["lock_until"]:
            st.error(f"帳號「{username}」已鎖定，請 {int(guard['lock_until'] - time.time()) + 1} 秒後再試")
        elif user and hash_password(password, user["salt"]) == user["hash"]:
            guard["fails"] = 0
            st.session_state["user"] = {"id": username, "name": user["name"], "role": user["role"]}
            st.rerun()
        else:
            guard["fails"] += 1
            if guard["fails"] >= MAX_FAIL:
                guard["lock_until"] = time.time() + LOCK_SECONDS
                guard["fails"] = 0
                st.error(f"帳號「{username}」連續輸錯 {MAX_FAIL} 次，鎖定 {LOCK_SECONDS} 秒")
            else:
                st.error(f"帳號或密碼錯誤（還可以再試 {MAX_FAIL - guard['fails']} 次）")

    st.info("**測試帳號**\n\n"
            "- 護理師：nurse1 / nurse111（王怡婷）、nurse2 / nurse222（陳雅雯）、nurse3 / nurse333（林志豪）\n"
            "- 管理員：admin / admin123（李明哲）")
    st.warning("⚠️ 這是共用的展示環境，你的領用、進貨操作其他人也看得到。")
    st.caption("本系統使用模擬資料，僅供學習展示")
    st.stop()


if "user" not in st.session_state:
    login_page()

user = st.session_state["user"]


# ---------- 側邊選單 ----------
st.sidebar.title("🏥 AI 智慧醫療耗材管理與需求預測系統")
st.sidebar.success(f"👤 {user['name']}（{user['role']}）")
page = st.sidebar.radio("功能選單", PAGES[user["role"]])
if st.sidebar.button("登出"):
    del st.session_state["user"]
    st.rerun()
st.sidebar.caption("本系統使用模擬資料，僅供學習展示")


# ---------- 庫存總覽頁 ----------
if page == "庫存總覽":
    st.title("📦 庫存總覽")
    overview = build_overview(items, usage)

    # 上方三個數字卡片
    c1, c2, c3 = st.columns(3)
    c1.metric("耗材種類", f"{len(overview)} 種")
    c2.metric("需立即補貨", f"{(overview['狀態'] == '🔴 需立即補貨').sum()} 項")
    c3.metric("需注意", f"{(overview['狀態'] == '🟡 注意').sum()} 項")

    # 庫存表格
    st.subheader("目前庫存")
    st.dataframe(overview, width="stretch", hide_index=True)

    # 用量趨勢圖
    st.subheader("用量趨勢")
    choice = st.selectbox("選擇耗材", overview["耗材"])
    days = st.slider("顯示最近幾天", min_value=30, max_value=365, value=90, step=30)
    data = usage[(usage["耗材"] == choice) & (usage["日期"] < taiwan_today())].tail(days).set_index("日期")["用量"]
    st.line_chart(data)
    st.caption("圖表只顯示到昨天；今天還沒過完，今天的領用量會在明天重設前列在操作紀錄中。")

# ---------- 領用與進貨頁 ----------
elif page == "領用與進貨":
    st.title("📝 領用與進貨")

    # 顯示上一次操作的結果
    if "msg" in st.session_state:
        ok, text = st.session_state.pop("msg")
        (st.success if ok else st.error)(text)

    col1, col2 = st.columns(2)
    with col1:
        item = st.selectbox("選擇耗材", items["耗材"])
        kinds = ["領用", "進貨"] if user["role"] == "管理員" else ["領用"]   # 只有管理員可以進貨
        kind = st.radio("類型", kinds, horizontal=True)
        qty = st.number_input("數量", min_value=1, value=10, step=1)

        row = items[items["耗材"] == item].iloc[0]
        st.caption(f"目前庫存：{row['目前庫存']} {row['單位']}　｜　安全庫存：{row['安全庫存']} {row['單位']}")

        if st.button(f"確認{kind}", type="primary"):
            st.session_state["msg"] = update_stock(item, kind, int(qty))
            st.rerun()

    with col2:
        st.subheader("最近操作紀錄")
        log = load_log()
        if log.empty:
            st.info("目前還沒有任何操作紀錄")
        else:
            st.dataframe(log.iloc[::-1].head(10), width="stretch", hide_index=True)

# ---------- 需求預測頁 ----------
elif page == "需求預測":
    st.title("🤖 AI 需求預測")
    st.write("使用 Prophet 時間序列模型，從過去的用量紀錄學習季節與星期的規律，預測未來的需求。")

    col1, col2 = st.columns(2)
    item = col1.selectbox("選擇耗材", items["耗材"])
    days = col2.slider("預測未來幾天", min_value=7, max_value=90, value=30, step=7)

    today = taiwan_today()
    history = usage[(usage["耗材"] == item) & (usage["日期"] < today)][["日期", "用量"]].reset_index(drop=True)  # 今天還沒過完，不拿來訓練
    with st.spinner("AI 模型訓練中，請稍候..."):
        forecast, mape = run_forecast(history, days)

    last_date = history["日期"].max()
    future = forecast[forecast["ds"] > last_date]
    future_total = int(future["yhat"].clip(lower=0).sum())

    row = items[items["耗材"] == item].iloc[0]
    stock = int(row["目前庫存"])
    cum = future["yhat"].clip(lower=0).cumsum()
    run_out = future.loc[cum > stock, "ds"]
    run_out_text = run_out.iloc[0].strftime("%Y-%m-%d") if len(run_out) else f"{days} 天內不會用完"

    # 數字卡片
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"未來 {days} 天預測用量", f"{future_total} {row['單位']}")
    c2.metric("目前庫存", f"{stock} {row['單位']}")
    c3.metric("預計用完日期", run_out_text)
    c4.metric("模型平均誤差 (MAPE)", f"{mape}%")

    # 預測圖：最近 180 天實際用量 + 未來預測
    st.subheader("實際用量與預測")
    chart = forecast[forecast["ds"] > last_date - pd.Timedelta(days=180)][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    chart = chart.merge(history.rename(columns={"日期": "ds", "用量": "實際用量"}), on="ds", how="left")
    chart = chart.rename(columns={"yhat": "AI 預測", "yhat_lower": "預測下限", "yhat_upper": "預測上限"}).set_index("ds")
    st.line_chart(chart[["實際用量", "AI 預測", "預測下限", "預測上限"]])
    st.caption("預測上下限代表 80% 的可能範圍；模型平均誤差是用最後 30 天實際資料測試的結果，越小越準。")

    # AI 學到的規律
    with st.expander("🔍 AI 從資料中學到了什麼？"):
        weekdays = ["一", "二", "三", "四", "五", "六", "日"]
        week = forecast.tail(7).copy()
        week["d"] = week["ds"].dt.day_of_week
        week = week.sort_values("d")
        week["星期"] = week["d"].map(lambda d: "星期" + weekdays[d])
        st.write("**星期的影響**（正數代表用量比平均多，負數代表比較少）")
        st.bar_chart(week.set_index("星期")["weekly"].round(1), sort=False)

        st.write("**季節的影響**（一年中每個月份的變化）")
        year = forecast[["ds", "yearly"]].copy()
        year["月份"] = year["ds"].dt.month
        st.bar_chart(year.groupby("月份")["yearly"].mean().round(1))

# ---------- 補貨提醒頁 ----------
elif page == "補貨提醒":
    st.title("🔔 補貨提醒")
    st.write("根據 AI 預測的未來用量，計算每項耗材預計用完的日期與建議補貨量。")

    col1, col2 = st.columns(2)
    lead = col1.slider("廠商交貨天數（下單後幾天會到貨）", min_value=1, max_value=21, value=7)
    plan = col2.slider("補貨要準備幾天的量", min_value=14, max_value=60, value=30, step=7)

    rows = []
    progress = st.progress(0, text="AI 正在預測各項耗材...")
    for i, it in items.iterrows():
        today = taiwan_today()
        history = usage[(usage["耗材"] == it["耗材"]) & (usage["日期"] < today)][["日期", "用量"]].reset_index(drop=True)  # 今天還沒過完，不拿來訓練
        forecast, _ = run_forecast(history, 90)
        future = forecast[forecast["ds"] > history["日期"].max()]
        daily = future["yhat"].clip(lower=0)

        stock = int(it["目前庫存"])
        cum = daily.cumsum()
        n = int((cum <= stock).sum())                                # 從資料最後一天起，庫存撐得過幾天
        if n < len(future):
            run_out_date = future["ds"].iloc[n]
            days_left = max(0, (run_out_date - taiwan_today()).days)  # 從今天算起還剩幾天
            run_out = run_out_date.strftime("%Y-%m-%d")
        else:
            days_left, run_out = 90, "90 天以上"

        need = daily.head(lead + plan).sum() + it["安全庫存"] - stock  # 等貨期間＋準備天數的用量＋安全庫存
        suggest = max(0, int(-(-need // 10) * 10))                    # 無條件進位到 10 的倍數

        if days_left <= lead:
            status = "🔴 立即下單"
        elif days_left <= lead + 7:
            status = "🟡 本週內下單"
        else:
            status = "🟢 暫不需要"

        rows.append({"耗材": it["耗材"], "單位": it["單位"], "目前庫存": stock,
                     "預計可用天數": days_left, "預計用完日期": run_out,
                     f"未來{plan}天預測用量": int(daily.head(plan).sum()),
                     "建議補貨量": suggest, "建議": status})
        progress.progress((i + 1) / len(items), text=f"AI 正在預測：{it['耗材']}")
    progress.empty()

    result = pd.DataFrame(rows).sort_values("預計可用天數")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 立即下單", f"{(result['建議'] == '🔴 立即下單').sum()} 項")
    c2.metric("🟡 本週內下單", f"{(result['建議'] == '🟡 本週內下單').sum()} 項")
    c3.metric("🟢 暫不需要", f"{(result['建議'] == '🟢 暫不需要').sum()} 項")

    st.subheader("補貨建議清單")
    st.dataframe(result, width="stretch", hide_index=True)
    st.caption("建議補貨量 ＝（交貨天數＋準備天數）的預測用量 ＋ 安全庫存 － 目前庫存（無條件進位到 10）")

    # 下載採購清單
    order = result[result["建議補貨量"] > 0][["耗材", "單位", "建議補貨量", "建議"]].copy()
    order["建議"] = order["建議"].str[2:]                              # 去掉表情符號，Excel 才不會顯示成怪符號
    st.download_button("📥 下載採購清單（CSV）", order.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"採購清單_{datetime.now(TZ):%Y%m%d}.csv", mime="text/csv")

# ---------- 操作紀錄頁（僅管理員） ----------
elif page == "操作紀錄":
    st.title("🗂️ 操作紀錄")
    st.write("記錄每一次領用與進貨的時間、操作者與數量，方便追查異常。")
    log = load_log()
    if log.empty:
        st.info("目前還沒有任何操作紀錄")
    else:
        c1, c2 = st.columns(2)
        who = c1.multiselect("篩選操作者", log["操作者"].dropna().unique())
        what = c2.multiselect("篩選耗材", log["耗材"].dropna().unique())
        view = log
        if who:
            view = view[view["操作者"].isin(who)]
        if what:
            view = view[view["耗材"].isin(what)]
        st.dataframe(view.iloc[::-1], width="stretch", hide_index=True)
        st.download_button("📥 下載操作紀錄（CSV）", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"操作紀錄_{datetime.now(TZ):%Y%m%d}.csv", mime="text/csv")

    # 重設資料（展示前或資料被弄亂時使用）
    st.divider()
    st.subheader("🔄 重設資料")
    st.write("把庫存與用量恢復成初始的模擬資料，並清空所有操作紀錄。")
    confirm = st.checkbox("我確定要重設所有資料（無法復原）")
    if st.button("重設資料", disabled=not confirm):
        generate()
        st.success("資料已重設！")
        st.rerun()

else:
    st.title(page)
    st.info("🚧 這個功能還在建置中，之後的步驟會完成。")

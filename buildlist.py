# ============================================================
# BUILDLIST - Loc ma du dieu kien, luu vn_stocks_full.txt
# Chay truoc weeklyscan.py
# Fix v2:
#   - Cho phep ma alphanumeric (PC1, VN30, ...) thay vi chi alpha
#   - Loc ma bi han che giao dich (< 3 ngay giao dich/tuan TB)
# ============================================================

import os

API_KEY = os.environ["VNSTOCK_API_KEY"]
TOKEN   = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

import asyncio
import logging
import threading
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Bot

VN_TZ  = timezone(timedelta(hours=7))
OUTPUT = "vn_stocks_full.txt"

# ============================================================
# DIEU KIEN LOC
# ============================================================
GIA_TOI_THIEU          = 2.0        # nghin dong -> 2,000d
VOL_TUAN_TOI_THIEU     = 500_000
TUAN_TINH_VOL          = 13
MAX_WORKERS            = 20

# Lo loc ma han che giao dich:
# Neu TB so ngay giao dich / tuan < nguong nay -> loai
# HOSE giao dich 5 ngay/tuan, HNX 5 ngay, UPCOM 5 ngay
# Ma binh thuong >= 4 ngay/tuan (tru nghi le)
# Ma bi han che 1 buoi/ngay -> ~2.5 ngay tuong duong
# Ma bi han che 1 ngay/tuan -> ~1 ngay
NGAY_GD_TUAN_TOI_THIEU = 3.5        # < 3.5 ngay GD/tuan = bi han che

EXCLUDE = {
    "E1VFVN30","FUEKIVFS","FUEMAV30","FUEMAVND",
    "FUESSV30","FUESSVFL","FUETCC50","FUEVFVND","FUEVN100"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

now_vn = lambda: datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M")

# ============================================================
# RATE LIMITER
# ============================================================
class RateLimiter:
    def __init__(self, max_calls=130, period=60.0):
        self.max_calls, self.period = max_calls, period
        self._lock, self._calls = threading.Lock(), []

    def acquire(self):
        while True:
            with self._lock:
                now = time.time()
                self._calls = [t for t in self._calls if now - t < self.period]
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                wait = self.period - (now - self._calls[0]) + 0.01
            time.sleep(max(wait, 0.05))

_rate_limiter = RateLimiter()

# ============================================================
# LAY DANH SACH MA THO
# ============================================================
def get_all_symbols():
    from vnstock import Vnstock
    stock = Vnstock(show_log=False).stock(symbol="ACB", source="KBS")
    df = stock.listing.symbols_by_exchange()
    df = df[df["exchange"].isin(["HOSE","HNX","UPCOM"])].copy()
    all_syms = df["symbol"].tolist()

    filtered = []
    for s in all_syms:
        # FIX: cho phep alphanumeric (PC1, VN30,...) thay vi chi alpha
        # Yeu cau: 2-5 ky tu, chi gom chu va so, khong co ky hieu dac biet
        if not (2 <= len(s) <= 5):
            continue
        if not s.isalnum():
            continue
        if s in EXCLUDE:
            continue
        filtered.append(s)

    filtered = list(dict.fromkeys(filtered))  # dedup giu thu tu
    logging.info("Tong tren san: %d | Sau loc alphanumeric<=5: %d", len(all_syms), len(filtered))
    return filtered

# ============================================================
# KIEM TRA TUNG MA
# ============================================================
def get_stock_stats(symbol):
    """
    Tra ve (avg_weekly_vol, last_close, avg_trading_days_per_week)
    hoac None neu khong lay duoc data.

    avg_trading_days_per_week: TB so ngay co giao dich trong 1 tuan.
    Ma binh thuong: ~4.5-5. Ma han che: < 3.5
    """
    from vnstock import Vnstock
    end   = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    start = (datetime.now(VN_TZ) - timedelta(days=TUAN_TINH_VOL * 7 + 14)).strftime("%Y-%m-%d")

    for source in ("KBS","MSN","VCI"):
        for attempt in range(3):
            _rate_limiter.acquire()
            try:
                stock = Vnstock(show_log=False).stock(symbol=symbol, source=source)
                df = stock.quote.history(start=start, end=end, interval="1D")
                if df is None or (hasattr(df, "empty") and df.empty):
                    break

                df.columns = [c.lower() for c in df.columns]
                if "volume" not in df.columns or "close" not in df.columns:
                    break

                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                    df = df.set_index("time")
                elif df.index.dtype != "datetime64[ns]":
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                # Bo qua hang volume = 0 (ngay khong giao dich thuc su)
                df_traded = df[df["volume"] > 0]
                if len(df_traded) < 4:
                    break

                last_close     = df["close"].iloc[-1]
                weekly_vol     = df_traded["volume"].resample("W-FRI").sum().dropna()
                if len(weekly_vol) < 4:
                    break
                avg_weekly_vol = weekly_vol.tail(TUAN_TINH_VOL).mean()

                # Dem so ngay GD thuc su moi tuan
                # (ngay co volume > 0 va close > 0)
                daily_traded   = (df["volume"] > 0).astype(int)
                weekly_days    = daily_traded.resample("W-FRI").sum().dropna()
                recent_weeks   = weekly_days.tail(TUAN_TINH_VOL)
                # Loai tuan co < 3 ngay giao dich vi co the la tuan nghi le
                normal_weeks   = recent_weeks[recent_weeks >= 3]
                if len(normal_weeks) < 4:
                    # Qua it tuan binh thuong -> coi nhu co van de
                    avg_days_per_week = recent_weeks.mean() if len(recent_weeks) > 0 else 0.0
                else:
                    avg_days_per_week = normal_weeks.mean()

                return avg_weekly_vol, last_close, avg_days_per_week

            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["rate limit","429","too many"]):
                    wait = 15 * (attempt + 1)
                    logging.warning("[%s/%s] rate limit, cho %ds...", symbol, source, wait)
                    time.sleep(wait)
                    continue
                break

    logging.warning("[%s] khong lay duoc data tu KBS/MSN/VCI", symbol)
    return None

# ============================================================
# MAIN
# ============================================================
async def main():
    async with Bot(token=TOKEN) as bot:

        try:
            logging.info("Dang lay danh sach ma...")
            all_symbols = get_all_symbols()
            total = len(all_symbols)
        except Exception as e:
            logging.error("Loi lay danh sach: %s", e)
            await bot.send_message(chat_id=CHAT_ID, text=f"Loi lay danh sach ma:\n{e}\n{now_vn()}")
            return

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode="HTML",
            text=(
                f"<b>BAT DAU BUILD DANH SACH MA</b>\n\n"
                f"Tong ma can kiem tra : <b>{total}</b>\n"
                f"Dieu kien loc :\n"
                f"  1. San HOSE / HNX / UPCOM\n"
                f"  2. Ma alphanumeric 2-5 ky tu (PC1, VN30 OK)\n"
                f"  3. Gia >= {int(GIA_TOI_THIEU*1000):,}d\n"
                f"  4. Volume tuan TB >= {VOL_TUAN_TOI_THIEU:,}\n"
                f"  5. So ngay GD/tuan >= {NGAY_GD_TUAN_TOI_THIEU} (loai ma bi han che)\n"
                f"Workers    : {MAX_WORKERS} threads\n"
                f"Rate limit : 130 req/phut\n"
                f"Fallback   : KBS -> MSN -> VCI\n"
                f"{now_vn()}"
            )
        )

        passed        = []
        removed_gia   = []
        removed_vol   = []
        removed_hanche = []
        no_data       = []
        done_cnt      = [0]
        lock          = threading.Lock()
        start_time    = time.time()

        def check_one(sym):
            stats = get_stock_stats(sym)
            with lock:
                done_cnt[0] += 1
                n = done_cnt[0]

                if stats is None:
                    no_data.append(sym)
                else:
                    avg_vol, last_close, avg_days = stats

                    if last_close < GIA_TOI_THIEU:
                        removed_gia.append(sym)
                    elif avg_vol < VOL_TUAN_TOI_THIEU:
                        removed_vol.append(sym)
                    elif avg_days < NGAY_GD_TUAN_TOI_THIEU:
                        # Ma bi han che giao dich
                        removed_hanche.append(sym)
                        logging.info("[%s] han che GD: %.1f ngay/tuan", sym, avg_days)
                    else:
                        passed.append(sym)

                if n % 60 == 0 or n == total:
                    elapsed = time.time() - start_time
                    speed   = n / elapsed * 60 if elapsed > 0 else 0
                    logging.info(
                        "Da kiem tra %d/%d (%.0f ma/phut) | Giu: %d | Loai gia: %d | Loai vol: %d | Han che: %d | No data: %d",
                        n, total, speed, len(passed), len(removed_gia), len(removed_vol), len(removed_hanche), len(no_data)
                    )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_one, sym): sym for sym in all_symbols}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error("[pool] %s", e)

        passed.sort()
        with open(OUTPUT, "w", encoding="utf-8") as f:
            f.write("\n".join(passed) + "\n")

        elapsed = time.time() - start_time
        logging.info("Da luu %d ma -> %s (%.0fs)", len(passed), OUTPUT, elapsed)

        # Gui danh sach ma bi han che (de tham khao)
        if removed_hanche:
            hanche_str = ", ".join(sorted(removed_hanche)[:50])
            if len(removed_hanche) > 50:
                hanche_str += f"... (+{len(removed_hanche)-50} ma)"
            await bot.send_message(
                chat_id=CHAT_ID, parse_mode="HTML",
                text=(
                    f"<b>MA BI HAN CHE GIAO DICH (da loai):</b>\n"
                    f"Tong: {len(removed_hanche)} ma\n"
                    f"{hanche_str}"
                )
            )

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode="HTML",
            text=(
                f"<b>BUILD DANH SACH HOAN TAT</b>\n\n"
                f"Tong ma ban dau            : <b>{total}</b>\n"
                f"Ma duoc giu lai            : <b>{len(passed)}</b>\n"
                f"Loai do gia &lt; 2,000d    : {len(removed_gia)}\n"
                f"Loai do vol &lt; 500,000   : {len(removed_vol)}\n"
                f"Loai do bi han che GD      : {len(removed_hanche)}\n"
                f"Loai do khong co data      : {len(no_data)}\n"
                f"File : {OUTPUT}\n"
                f"Thoi gian: {elapsed:.0f}s\n"
                f"{now_vn()}"
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
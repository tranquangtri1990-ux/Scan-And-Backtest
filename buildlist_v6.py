# ============================================================
# BUILDLIST v6 - Lọc mã đủ điều kiện, lưu vn_stocks_full.txt
# Chạy trước weeklyscan_v6.py
# vnstock 6 + nguồn TCBS (Smart OTP qua Telegram: /otp 123456)
#   - Danh sách mã: bảng room ngoại TCBS theo sàn HOSE/HNX/UPCOM
#     (Reference.equity.list chỉ có ở dnse/ssi, bạn chưa có khóa)
#   - Nếu lấy danh sách lỗi/quá ít mã -> dùng lại vn_stocks_full.txt cũ
# ============================================================

import os

TOKEN   = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Bot

import vn6_common as vc

VN_TZ  = timezone(timedelta(hours=7))
OUTPUT = "vn_stocks_full.txt"

# ============================================================
# ĐIỀU KIỆN LỌC (giữ nguyên bản cũ)
# ============================================================
GIA_TOI_THIEU          = 2.0        # nghìn đồng -> 2,000đ
VOL_TUAN_TOI_THIEU     = 500_000
TUAN_TINH_VOL          = 13
MAX_WORKERS            = int(os.environ.get("MAX_WORKERS", "10"))
NGAY_GD_TUAN_TOI_THIEU = 3.5        # < 3.5 ngày GD/tuần = bị hạn chế
MIN_UNIVERSE           = 500        # ít hơn số này coi như lấy danh sách lỗi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
now_vn = lambda: datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M")


def read_old_list():
    try:
        with open(OUTPUT, "r", encoding="utf-8") as f:
            return vc.clean_symbols([l.strip() for l in f if l.strip()])
    except Exception:
        return []


def get_stock_stats(symbol):
    """(avg_weekly_vol, last_close, avg_trading_days_per_week) hoặc None."""
    start = (datetime.now(VN_TZ) - timedelta(days=TUAN_TINH_VOL * 7 + 14)).strftime("%Y-%m-%d")
    for source in vc.DATA_SOURCES:
        for attempt in range(3):
            try:
                df = vc.fetch_daily(symbol, start, source)
                if df is None:
                    break
                df_traded = df[df["Volume"] > 0]
                if len(df_traded) < 4:
                    break
                last_close = df["Close"].iloc[-1]
                weekly_vol = df_traded["Volume"].resample("W-FRI").sum().dropna()
                if len(weekly_vol) < 4:
                    break
                avg_weekly_vol = weekly_vol.tail(TUAN_TINH_VOL).mean()

                weekly_days  = (df["Volume"] > 0).astype(int).resample("W-FRI").sum().dropna()
                recent_weeks = weekly_days.tail(TUAN_TINH_VOL)
                normal_weeks = recent_weeks[recent_weeks >= 3]   # bỏ tuần nghỉ lễ
                if len(normal_weeks) < 4:
                    avg_days = recent_weeks.mean() if len(recent_weeks) > 0 else 0.0
                else:
                    avg_days = normal_weeks.mean()
                return avg_weekly_vol, last_close, avg_days
            except vc.SessionExpired:
                raise
            except Exception as e:
                if vc.is_rate_limited(e):
                    time.sleep(15 * (attempt + 1))
                    continue
                break
    logging.warning("[%s] không lấy được dữ liệu", symbol)
    return None


async def main():
    async with Bot(token=TOKEN) as bot:
        # 1) Phiên TCBS
        if not await vc.ensure_tcbs_session(bot, CHAT_ID):
            raise SystemExit(1)

        # 2) Danh sách mã
        loop = asyncio.get_running_loop()
        try:
            all_symbols = await loop.run_in_executor(None, vc.list_symbols_tcbs)
        except vc.SessionExpired:
            await bot.send_message(chat_id=CHAT_ID, text="❌ Phiên TCBS hết hạn khi lấy danh sách. Chạy lại.")
            raise SystemExit(1)
        source_note = "bảng room ngoại TCBS"
        if len(all_symbols) < MIN_UNIVERSE:
            old = read_old_list()
            if old:
                all_symbols = old
                source_note = f"file {OUTPUT} cũ (TCBS chỉ trả < {MIN_UNIVERSE} mã)"
            elif not all_symbols:
                await bot.send_message(chat_id=CHAT_ID, text="❌ Không lấy được danh sách mã và không có file cũ.")
                raise SystemExit(1)
        total = len(all_symbols)

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode="HTML",
            text=(f"<b>BẮT ĐẦU BUILD DANH SÁCH MÃ (vnstock 6)</b>\n\n"
                  f"Tổng mã cần kiểm tra : <b>{total}</b> ({source_note})\n"
                  f"Điều kiện lọc :\n"
                  f"  1. Giá &gt;= {int(GIA_TOI_THIEU*1000):,}đ\n"
                  f"  2. Volume tuần TB &gt;= {VOL_TUAN_TOI_THIEU:,}\n"
                  f"  3. Số ngày GD/tuần &gt;= {NGAY_GD_TUAN_TOI_THIEU}\n"
                  f"Workers: {MAX_WORKERS} | Rate: {vc.rate_limiter.max_calls}/phút\n"
                  f"{now_vn()}"))

        passed, removed_gia, removed_vol, removed_hanche, no_data = [], [], [], [], []
        done_cnt, lock, start_time = [0], threading.Lock(), time.time()

        def check_one(sym):
            try:
                stats = get_stock_stats(sym)
            except vc.SessionExpired:
                return
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
                        removed_hanche.append(sym)
                    else:
                        passed.append(sym)
                if n % 60 == 0 or n == total:
                    el = time.time() - start_time
                    logging.info("Đã kiểm tra %d/%d (%.0f mã/phút) | Giữ %d | giá %d | vol %d | hạn chế %d | no data %d",
                                 n, total, n / el * 60 if el else 0, len(passed),
                                 len(removed_gia), len(removed_vol), len(removed_hanche), len(no_data))

        def run_pool():
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(check_one, s) for s in all_symbols]
                for f in as_completed(futs):
                    try:
                        f.result()
                    except Exception as e:
                        logging.error("[pool] %s", e)

        await loop.run_in_executor(None, run_pool)

        if vc.SESSION_DEAD.is_set():
            await bot.send_message(chat_id=CHAT_ID,
                                   text="❌ Phiên TCBS hết hạn giữa chừng, KHÔNG ghi đè danh sách. Chạy lại.")
            raise SystemExit(1)

        elapsed = time.time() - start_time
        # An toàn: nếu quá nhiều mã không có dữ liệu thì không ghi đè file cũ
        if len(no_data) > 0.5 * total or len(passed) < 100:
            await bot.send_message(
                chat_id=CHAT_ID,
                text=(f"⚠️ Kết quả bất thường (giữ {len(passed)}, không dữ liệu {len(no_data)}/{total}). "
                      f"Không ghi đè {OUTPUT}."))
            raise SystemExit(1)

        passed.sort()
        with open(OUTPUT, "w", encoding="utf-8") as f:
            f.write("\n".join(passed) + "\n")
        logging.info("Đã lưu %d mã -> %s (%.0fs)", len(passed), OUTPUT, elapsed)

        if removed_hanche:
            hs = ", ".join(sorted(removed_hanche)[:50])
            if len(removed_hanche) > 50:
                hs += f"... (+{len(removed_hanche)-50} mã)"
            await bot.send_message(chat_id=CHAT_ID, parse_mode="HTML",
                                   text=f"<b>MÃ BỊ HẠN CHẾ GIAO DỊCH (đã loại):</b>\nTổng: {len(removed_hanche)}\n{hs}")

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode="HTML",
            text=(f"<b>BUILD DANH SÁCH HOÀN TẤT</b>\n\n"
                  f"Tổng mã ban đầu       : <b>{total}</b>\n"
                  f"Mã được giữ lại       : <b>{len(passed)}</b>\n"
                  f"Loại do giá &lt; 2,000đ  : {len(removed_gia)}\n"
                  f"Loại do vol &lt; 500,000 : {len(removed_vol)}\n"
                  f"Loại do hạn chế GD    : {len(removed_hanche)}\n"
                  f"Loại do không có data : {len(no_data)}\n"
                  f"File: {OUTPUT} | {elapsed:.0f}s\n{now_vn()}"))


if __name__ == "__main__":
    asyncio.run(main())

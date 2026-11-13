# ============================================================
# WEEKLY SCAN v6 - Chạy độc lập, không polling
# Chạy xong tự thoát, xuất TXT, thông báo Telegram
# vnstock thế hệ 6: Market("tcbs").equity(sym).ohlcv(...)
# Mở phiên TCBS bằng Smart OTP qua Telegram:  /otp 123456
# ============================================================

import os

TOKEN   = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

import asyncio
import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Bot

import vn6_common as vc

VN_TZ = timezone(timedelta(hours=7))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
WEEKLY_PROGRESS = 100

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
now_vn = lambda: datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M')


# ============================================================
# ĐỌC DANH SÁCH MÃ
# ============================================================
def get_all_symbols(filename='vn_stocks_full.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw = [line.strip() for line in f if line.strip()]
        filtered = vc.clean_symbols(raw)
        logging.info('[symbols] Đọc %d dòng → %d mã hợp lệ', len(raw), len(filtered))
        return filtered
    except Exception as e:
        logging.error('[symbols] Lỗi đọc file: %s', e)
        return []


# ============================================================
# CHỈ BÁO KỸ THUẬT (giữ nguyên)
# ============================================================
def smma(series, period):
    values = series.values.astype(float)
    result = np.full(len(values), np.nan)
    count, start = 0, -1
    for i, v in enumerate(values):
        if not np.isnan(v):
            count += 1
            if count == period:
                start = i
                break
        else:
            count = 0
    if start == -1:
        return pd.Series(result, index=series.index)
    result[start] = np.mean(values[start - period + 1: start + 1])
    for i in range(start + 1, len(values)):
        result[i] = result[i-1] if np.isnan(values[i]) else (result[i-1] * (period-1) + values[i]) / period
    return pd.Series(result, index=series.index)


def calc_indicators(weekly):
    df = weekly.copy()
    df['ma20_vol'] = df['Volume'].rolling(20).mean()
    delta = df['Close'].diff()
    df['rsi'] = 100 - (100 / (1 + smma(delta.where(delta > 0, 0.0), 14) /
                                    smma((-delta).where(delta < 0, 0.0), 14)))
    df['sma_rsi'] = df['rsi'].rolling(14).mean()
    return df


# ============================================================
# WEEKLY SIGNAL
# ============================================================
def check_weekly_signal(symbol):
    if vc.SESSION_DEAD.is_set():
        return None
    try:
        start_date = (datetime.now(VN_TZ) - timedelta(days=500)).strftime('%Y-%m-%d')
        daily, weekly = vc.get_data(symbol, start_date=start_date)
        if daily is None or weekly is None or len(weekly) < 30:
            return None
        vol = weekly['Volume'].iloc[-1]
        if vol <= 500_000:
            return None
        df_w = calc_indicators(weekly)
        r1, r2 = df_w['rsi'].iloc[-1], df_w['rsi'].iloc[-2]
        s1, s2 = df_w['sma_rsi'].iloc[-1], df_w['sma_rsi'].iloc[-2]
        if any(np.isnan(v) for v in [r1, r2, s1, s2]):
            return None
        if r2 <= s2 and r1 > s1:
            logging.info('✅ [%s] Vol=%.0fK | RSI: %.1f→%.1f | SMA: %.1f→%.1f',
                         symbol, vol/1000, r2, r1, s2, s1)
            return {
                'symbol': symbol, 'week': weekly.index[-1].strftime('%Y-%m-%d'),
                'close': round(weekly['Close'].iloc[-1], 2), 'volume': int(vol),
                'rsi': round(r1, 2), 'sma_rsi': round(s1, 2)
            }
    except vc.SessionExpired:
        return None
    except Exception as e:
        logging.warning('❌ [%s] %s', symbol, str(e)[:80])
    return None


def run_pool_sync(fn, symbols, max_workers=MAX_WORKERS):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error('[pool] %s', e)


# ============================================================
# MAIN
# ============================================================
async def main():
    symbols = get_all_symbols()
    total = len(symbols)

    async with Bot(token=TOKEN) as bot:
        if total == 0:
            await bot.send_message(chat_id=CHAT_ID,
                                   text='❌ Không tìm thấy vn_stocks_full.txt hoặc file rỗng.\nChạy buildlist trước.')
            raise SystemExit(1)

        # Mở phiên TCBS (hỏi OTP qua Telegram nếu cần)
        if not await vc.ensure_tcbs_session(bot, CHAT_ID):
            raise SystemExit(1)

        start_time = time.time()
        await bot.send_message(
            chat_id=CHAT_ID, parse_mode='HTML',
            text=(f"<b>📅 BẮT ĐẦU WEEKLY SCAN (vnstock 6)</b>\n\n"
                  f"Tổng số mã : <b>{total}</b>\n"
                  f"Điều kiện  :\n"
                  f"  1. Volume tuần &gt; 500,000\n"
                  f"  2. RSI(14) cắt lên SMA(RSI,14)\n"
                  f"Workers    : {MAX_WORKERS} threads\n"
                  f"Rate limit : {vc.rate_limiter.max_calls} lần gọi/phút\n\n"
                  f"Cập nhật mỗi {WEEKLY_PROGRESS} mã...\n🕐 {now_vn()}"))

        results, done_cnt = [], [0]
        lock = threading.Lock()
        progress_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def scan_one(sym):
            res = check_weekly_signal(sym)
            with lock:
                done_cnt[0] += 1
                n = done_cnt[0]
                if res:
                    results.append(res)
                if n % WEEKLY_PROGRESS == 0 or n == total:
                    elapsed = time.time() - start_time
                    remain = (elapsed / n) * (total - n) if n > 0 else 0
                    speed = n / elapsed * 60 if elapsed > 0 else 0
                    msg = (f"📊 <b>TIẾN TRÌNH WEEKLY SCAN</b>\n\n"
                           f"Đã xong : {n}/{total} ({n/total*100:.1f}%)\n"
                           f"✅ Tìm thấy: {len(results)} mã\n\n"
                           f"⏱ {elapsed:.0f}s | Còn ~{remain:.0f}s\n🚀 {speed:.0f} mã/phút")
                    asyncio.run_coroutine_threadsafe(progress_queue.put(msg), loop)

        async def report_progress():
            while True:
                msg = await progress_queue.get()
                if msg is None:
                    break
                try:
                    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='HTML')
                except Exception as e:
                    logging.warning('[progress] %s', e)

        def run_and_signal():
            run_pool_sync(scan_one, symbols)
            asyncio.run_coroutine_threadsafe(progress_queue.put(None), loop)

        await asyncio.gather(loop.run_in_executor(None, run_and_signal), report_progress())

        total_elapsed = time.time() - start_time

        if vc.SESSION_DEAD.is_set():
            await bot.send_message(chat_id=CHAT_ID,
                                   text='❌ Phiên TCBS hết hạn giữa chừng, kết quả không đầy đủ. Chạy lại và nhập OTP mới.')
            raise SystemExit(1)

        results.sort(key=lambda x: x['symbol'])
        logging.info('WEEKLY SCAN SUMMARY: %d/%d tín hiệu | %.0fs', len(results), total, total_elapsed)

        # Xuất TXT (workflow upload artifact ket_qua_weekly_*.txt)
        txt_name = f"ket_qua_weekly_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M')}.txt"
        with open(txt_name, 'w', encoding='utf-8') as f:
            f.write("symbol\tweek\tclose\tvolume\trsi\tsma_rsi\n")
            for r in results:
                f.write(f"{r['symbol']}\t{r['week']}\t{r['close']}\t{r['volume']}\t{r['rsi']}\t{r['sma_rsi']}\n")

        if results:
            header = (f"<b>📊 KẾT QUẢ WEEKLY SCAN</b>\n\n"
                      f"✅ Thỏa điều kiện: <b>{len(results)}/{total}</b> mã\n"
                      f"⏱ {total_elapsed:.0f}s | {total/total_elapsed*60:.0f} mã/phút\n🕐 {now_vn()}\n\n")
            body = "".join(f"{i}. {r['symbol']}\n" for i, r in enumerate(results, start=1))
            full_msg = header + body
            LIMIT = 4096
            if len(full_msg) <= LIMIT:
                await bot.send_message(chat_id=CHAT_ID, text=full_msg, parse_mode='HTML')
            else:
                await bot.send_message(chat_id=CHAT_ID, text=header, parse_mode='HTML')
                chunk = ""
                for line in body.splitlines(keepends=True):
                    if len(chunk) + len(line) > LIMIT:
                        await bot.send_message(chat_id=CHAT_ID, text=chunk)
                        chunk = ""
                    chunk += line
                if chunk:
                    await bot.send_message(chat_id=CHAT_ID, text=chunk)
        else:
            await bot.send_message(chat_id=CHAT_ID, parse_mode='HTML',
                                   text=(f"😔 Không tìm thấy mã nào thỏa điều kiện.\n"
                                         f"Tổng quét: {total}\n⏱ {total_elapsed:.0f}s\n🕐 {now_vn()}"))

        await bot.send_message(chat_id=CHAT_ID, text=f"✅ Weekly scan hoàn tất. Thoát.\n🕐 {now_vn()}")


if __name__ == '__main__':
    asyncio.run(main())

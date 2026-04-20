# ============================================================
# BUILDLIST - Lọc mã đủ điều kiện, lưu vn_stocks_full.txt
# Chạy trước weeklyscan.py
# ============================================================

import os

# Đọc từ biến môi trường (GitHub Secrets) — KHÔNG hardcode ở đây
API_KEY = os.environ['VNSTOCK_API_KEY']
TOKEN   = os.environ['TELEGRAM_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

import asyncio
import logging
import threading
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Bot

VN_TZ  = timezone(timedelta(hours=7))
OUTPUT = 'vn_stocks_full.txt'

# ============================================================
# ĐIỀU KIỆN LỌC
# ============================================================
GIA_TOI_THIEU       = 2.0        # đơn vị nghìn đồng → 2,000đ
VOL_TUAN_TOI_THIEU  = 500_000
TUAN_TINH_VOL       = 13
MAX_WORKERS         = 20

EXCLUDE = {
    'E1VFVN30', 'FUEKIVFS', 'FUEMAV30', 'FUEMAVND',
    'FUESSV30', 'FUESSVFL', 'FUETCC50', 'FUEVFVND', 'FUEVN100'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

now_vn = lambda: datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M')

# ============================================================
# RATE LIMITER - 130 req/phút
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
# LẤY DANH SÁCH MÃ THÔ
# ============================================================
def get_all_symbols():
    from vnstock import Vnstock
    stock = Vnstock(show_log=False).stock(symbol='ACB', source='KBS')
    df = stock.listing.symbols_by_exchange()
    df = df[df['exchange'].isin(['HOSE', 'HNX', 'UPCOM'])].copy()
    all_syms = df['symbol'].tolist()
    filtered = [
        s for s in all_syms
        if len(s) <= 3 and s.isalpha() and s not in EXCLUDE
    ]
    logging.info('Tổng trên sàn: %d | Sau lọc ≤3 ký tự: %d', len(all_syms), len(filtered))
    return filtered

# ============================================================
# KIỂM TRA TỪNG MÃ - Fallback KBS → MSN → VCI
# ============================================================
def get_stock_stats(symbol):
    from vnstock import Vnstock
    end   = datetime.now(VN_TZ).strftime('%Y-%m-%d')
    start = (datetime.now(VN_TZ) - timedelta(days=TUAN_TINH_VOL * 7 + 14)).strftime('%Y-%m-%d')

    for source in ('KBS', 'MSN', 'VCI'):
        for attempt in range(3):
            _rate_limiter.acquire()
            try:
                stock = Vnstock(show_log=False).stock(symbol=symbol, source=source)
                df = stock.quote.history(start=start, end=end, interval='1D')
                if df is None or (hasattr(df, 'empty') and df.empty):
                    break
                df.columns = [c.lower() for c in df.columns]
                if 'volume' not in df.columns or 'close' not in df.columns:
                    break
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                elif df.index.dtype != 'datetime64[ns]':
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                last_close     = df['close'].iloc[-1]
                weekly_vol     = df['volume'].resample('W-FRI').sum().dropna()
                if len(weekly_vol) < 4:
                    break
                avg_weekly_vol = weekly_vol.tail(TUAN_TINH_VOL).mean()
                return avg_weekly_vol, last_close
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ['rate limit', '429', 'too many']):
                    wait = 15 * (attempt + 1)
                    logging.warning('[%s/%s] rate limit, cho %ds...', symbol, source, wait)
                    time.sleep(wait)
                    continue
                break

    logging.warning('[%s] khong lay duoc data tu KBS/MSN/VCI', symbol)
    return None

# ============================================================
# MAIN
# ============================================================
async def main():
    async with Bot(token=TOKEN) as bot:

        try:
            logging.info('Đang lấy danh sách mã từ KBS...')
            all_symbols = get_all_symbols()
            total = len(all_symbols)
        except Exception as e:
            logging.error('Lỗi lấy danh sách: %s', e)
            await bot.send_message(chat_id=CHAT_ID,
                text=f"Loi lay danh sach ma:\n{e}\n{now_vn()}")
            return

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode='HTML',
            text=(
                f"<b>BAT DAU BUILD DANH SACH MA</b>\n\n"
                f"Tong ma can kiem tra : <b>{total}</b>\n"
                f"Dieu kien loc :\n"
                f"  1. San HOSE / HNX / UPCOM\n"
                f"  2. Ma 3 ky tu\n"
                f"  3. Gia &gt;= {int(GIA_TOI_THIEU*1000):,}d\n"
                f"  4. Volume tuan TB &gt;= {VOL_TUAN_TOI_THIEU:,}\n"
                f"Workers    : {MAX_WORKERS} threads\n"
                f"Rate limit : 130 req/phut\n"
                f"Fallback   : KBS -> MSN -> VCI\n"
                f"{now_vn()}"
            )
        )

        passed, removed_gia, removed_vol, no_data = [], [], [], []
        done_cnt = [0]
        lock = threading.Lock()
        start_time = time.time()

        def check_one(sym):
            stats = get_stock_stats(sym)
            with lock:
                done_cnt[0] += 1
                n = done_cnt[0]

                if stats is None:
                    no_data.append(sym)
                else:
                    avg_vol, last_close = stats
                    if last_close < GIA_TOI_THIEU:
                        removed_gia.append(sym)
                    elif avg_vol < VOL_TUAN_TOI_THIEU:
                        removed_vol.append(sym)
                    else:
                        passed.append(sym)

                if n % 60 == 0 or n == total:
                    elapsed = time.time() - start_time
                    speed = n / elapsed * 60 if elapsed > 0 else 0
                    logging.info(
                        'Da kiem tra %d/%d (%.0f ma/phut) | Giu: %d | Loai gia: %d | Loai vol: %d | No data: %d',
                        n, total, speed, len(passed), len(removed_gia), len(removed_vol), len(no_data)
                    )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_one, sym): sym for sym in all_symbols}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error('[pool] %s', e)

        passed.sort()
        with open(OUTPUT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(passed) + '\n')

        elapsed = time.time() - start_time
        logging.info('Da luu %d ma -> %s (%.0fs)', len(passed), OUTPUT, elapsed)

        await bot.send_message(
            chat_id=CHAT_ID, parse_mode='HTML',
            text=(
                f"<b>BUILD DANH SACH HOAN TAT</b>\n\n"
                f"Tong ma ban dau          : <b>{total}</b>\n"
                f"Ma duoc giu lai          : <b>{len(passed)}</b>\n"
                f"Loai do gia &lt; 2,000d  : {len(removed_gia)}\n"
                f"Loai do vol &lt; 500,000 : {len(removed_vol)}\n"
                f"Loai do khong co data    : {len(no_data)}\n"
                f"File : {OUTPUT}\n"
                f"Thoi gian: {elapsed:.0f}s\n"
                f"{now_vn()}"
            )
        )

if __name__ == '__main__':
    asyncio.run(main())

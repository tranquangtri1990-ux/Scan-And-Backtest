# ============================================================
# WEEKLY SCAN - Chạy độc lập, không polling
# Chạy xong tự thoát, xuất TXT, thông báo Telegram
# ============================================================

import os

# Đọc từ biến môi trường (GitHub Secrets) — KHÔNG hardcode ở đây
API_KEY = os.environ['VNSTOCK_API_KEY']
TOKEN   = os.environ['TELEGRAM_TOKEN']
CHAT_ID = os.environ['TELEGRAM_CHAT_ID']

import asyncio
import logging
import threading
import re
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Bot

VN_TZ = timezone(timedelta(hours=7))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
WEEKLY_PROGRESS = 100

now_vn = lambda: datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M')

# ============================================================
# RATE LIMITER
# ============================================================
class RateLimiter:
    def __init__(self, max_calls=300, period=60.0):
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
# VNSTOCK
# ============================================================
_Vnstock, _vnstock_lock = None, threading.Lock()

def get_vnstock_class():
    global _Vnstock
    if _Vnstock is None:
        with _vnstock_lock:
            if _Vnstock is None:
                from vnstock import Vnstock
                _Vnstock = Vnstock
    return _Vnstock

# ============================================================
# ĐỌC DANH SÁCH MÃ
# ============================================================
def get_all_symbols(filename='vn_stocks_full.txt'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw = [line.strip().upper() for line in f if line.strip()]

        # Giữ mã bắt đầu bằng chữ cái, tiếp theo là chữ/số, độ dài 2-5 ký tự
        # Hợp lệ: AAA, PC1, TV1, VC2, HT1, NT2, C4G ... | Loại: FUEVN100, ký tự đặc biệt
        symbols = [s for s in raw if re.fullmatch(r'[A-Z][A-Z0-9]{1,4}', s)]

        # Loại ETF và các mã đặc biệt đã biết
        exclude = {
            'E1VFVN30', 'FUEKIVFS', 'FUEMAV30', 'FUEMAVND',
            'FUESSV30', 'FUESSVFL', 'FUETCC50', 'FUEVFVND', 'FUEVN100'
        }
        filtered = [s for s in dict.fromkeys(symbols) if s not in exclude]

        logging.info('[symbols] Đọc %d dòng → %d mã hợp lệ (đã lọc mã có số)',
                     len(raw), len(filtered))
        return filtered
    except Exception as e:
        logging.error('[symbols] Lỗi đọc file: %s', e)
        return []

# ============================================================
# LẤY DỮ LIỆU GIÁ
# ============================================================
def _fetch_df(symbol, source, start_date='2022-01-01'):
    Vnstock = get_vnstock_class()
    _rate_limiter.acquire()
    stock = Vnstock(show_log=False).stock(symbol=symbol, source=source)
    end   = datetime.now(VN_TZ).strftime('%Y-%m-%d')
    raw   = stock.quote.history(start=start_date, end=end, interval='1D')

    df = pd.DataFrame(raw['data']) if isinstance(raw, dict) and 'data' in raw else raw
    if df is None or (hasattr(df, 'empty') and df.empty):
        return None

    df.columns = [c.lower() for c in df.columns]
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    elif df.index.dtype != 'datetime64[ns]':
        df.index = pd.to_datetime(df.index)

    rename_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
    df = df.rename(columns=rename_map).sort_index()
    req_cols = ['Close', 'Volume']
    if not all(c in df.columns for c in req_cols):
        return None
    return df.dropna(subset=req_cols) if not df.empty else None

def get_data(symbol, start_date='2022-01-01'):
    last_errors = []
    for source in ('KBS', 'VCI', 'MSN'):
        try:
            df = _fetch_df(symbol, source, start_date)
            if df is not None:
                weekly = df.resample('W-FRI').agg({'Close': 'last', 'Volume': 'sum'}).dropna()
                return df, weekly
            last_errors.append(f"{source}:empty")
        except Exception as e:
            err = str(e)
            logging.warning('[get_data] %s/%s: %s', symbol, source, err[:120])
            last_errors.append(f"{source}:{err[:120]}")
            if any(k in err.lower() for k in ['rate limit', '429', 'too many', 'exceeded']):
                time.sleep(30)
                try:
                    df2 = _fetch_df(symbol, source, start_date)
                    if df2 is not None:
                        weekly = df2.resample('W-FRI').agg({'Close': 'last', 'Volume': 'sum'}).dropna()
                        return df2, weekly
                except:
                    pass
    return None, last_errors

# ============================================================
# CHỈ BÁO KỸ THUẬT
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
    try:
        start_date = (datetime.now(VN_TZ) - timedelta(days=500)).strftime('%Y-%m-%d')
        daily, weekly = get_data(symbol, start_date=start_date)
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
    except Exception as e:
        logging.warning('❌ [%s] %s', symbol, str(e)[:80])
    return None

# ============================================================
# P/E & CỔ TỨC (chỉ gọi cho các mã đã thỏa điều kiện, sau khi quét xong)
# ------------------------------------------------------------
# P/E   : Finance(source='KBS').ratio() -> tidy format (item/item_id + cột kỳ),
#         lấy dòng item_id == 'pe', cột kỳ mới nhất.
# Cổ tức: Company(source='TCBS').events() -> KHÔNG có cột số tách sẵn, phải
#         parse "tỷ lệ xx%" từ event_title (vd "Trả cổ tức bằng tiền tỷ lệ 20%",
#         "Phát hành cổ phiếu trả cổ tức tỷ lệ 27.6%"). Đây là cách đáng tin cậy
#         nhất theo mẫu tiêu đề thực tế của vnstock, nhưng vẫn nên chạy thử
#         get_fundamentals('BCM') 1 lần và đối chiếu với tin tức cổ tức thật của
#         mã đó trước khi tin tưởng hoàn toàn, vì công ty có thể đặt tiêu đề khác kiểu.
# ============================================================
PAR_VALUE = 10_000  # mệnh giá cổ phiếu VN, dùng để quy đổi % cổ tức tiền mặt ra VNĐ/cp

def _first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _latest_pe(ratio_df):
    """finance.ratio() (nguồn KBS) trả dạng tidy: cột item/item_id + các cột kỳ báo cáo
    (vd '2025-Q4','2025-Q3',...), kỳ mới nhất nằm ở cột đầu tiên sau item_id."""
    if ratio_df is None or ratio_df.empty or 'item_id' not in ratio_df.columns:
        return None
    row = ratio_df[ratio_df['item_id'].astype(str).str.lower() == 'pe']
    if row.empty:
        return None
    period_cols = [c for c in ratio_df.columns if c not in ('item', 'item_id')]
    if not period_cols:
        return None
    val = row.iloc[0][period_cols[0]]
    return float(val) if pd.notna(val) else None

_PCT_RE = re.compile(r'tỷ lệ\s*([\d.,]+)\s*%', re.IGNORECASE)

def _parse_dividend_events(events_df):
    """company.events() (nguồn TCBS) trả tiêu đề dạng câu, vd:
    'VCB - Phát hành cổ phiếu trả cổ tức tỷ lệ 27.6%' (cổ phiếu)
    'ABC - Trả cổ tức bằng tiền tỷ lệ 20%' (tiền mặt)
    Hàm này parse tỷ lệ % + năm từ text, không có cột số sẵn nên đây là suy đoán
    tốt nhất theo mẫu tiêu đề — nên kiểm tra lại nếu công ty đặt tiêu đề khác kiểu.
    Trả về (cash, stock), mỗi cái là (year, value) hoặc None."""
    if events_df is None or events_df.empty:
        return None, None
    df = events_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    title_col = _first_col(df, ['event_title', 'title'])
    date_col = _first_col(df, ['record_date', 'public_date', 'exright_date'])
    if title_col is None:
        return None, None
    if date_col:
        df = df.sort_values(date_col, ascending=False)

    cash = stock = None
    for _, row in df.iterrows():
        title = str(row.get(title_col, ''))
        low = title.lower()
        if 'cổ tức' not in low and 'dividend' not in low:
            continue
        m = _PCT_RE.search(title)
        if not m:
            continue
        pct = float(m.group(1).replace(',', '.'))
        year = str(row.get(date_col, ''))[:4] if date_col else ''
        if 'cổ phiếu' in low and stock is None:
            stock = (year, pct)
        elif 'tiền' in low and cash is None:
            cash = (year, round(pct / 100 * PAR_VALUE))
        if cash and stock:
            break
    return cash, stock

def get_fundamentals(symbol):
    """Trả về dict: pe (float|None), cash (year, amount_vnd)|None, stock (year, pct)|None."""
    out = {'pe': None, 'cash': None, 'stock': None}
    Vnstock = get_vnstock_class()

    # --- P/E hiện tại: chỉ số tài chính, nguồn KBS (khuyến nghị) ---
    try:
        _rate_limiter.acquire()
        stock = Vnstock(show_log=False).stock(symbol=symbol, source='KBS')
        ratio = stock.finance.ratio(period='quarter')
        out['pe'] = _latest_pe(ratio)
    except Exception as e:
        logging.warning('[fund/pe] %s: %s', symbol, str(e)[:100])

    # --- Cổ tức gần nhất (tiền mặt + cổ phiếu): sự kiện doanh nghiệp, nguồn TCBS ---
    try:
        _rate_limiter.acquire()
        stock = Vnstock(show_log=False).stock(symbol=symbol, source='TCBS')
        events = stock.company.events()
        out['cash'], out['stock'] = _parse_dividend_events(events)
    except Exception as e:
        logging.warning('[fund/div] %s: %s', symbol, str(e)[:100])

    return out

def format_result_line(idx, r, fund):
    sym = r['symbol']
    pe_str = f"{fund['pe']:.1f}" if fund.get('pe') is not None else ''
    cash_str = ''
    if fund.get('cash'):
        year, amount = fund['cash']
        cash_str = f"{year} {amount:,}".replace(',', '.') + "/cp"
    stock_str = ''
    if fund.get('stock'):
        year, pct = fund['stock']
        stock_str = f"{year} {pct:.0f}%"

    line = f"{idx}. {sym:<8}P/E: {pe_str:<8}"
    if cash_str:
        line += f"CTTM: {cash_str:<16}"
    if stock_str:
        line += f"CTCP: {stock_str}"
    return line.rstrip()

# ============================================================
# POOL
# ============================================================
def run_pool_sync(fn, items, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, item): item for item in items}
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
            await bot.send_message(
                chat_id=CHAT_ID,
                text='❌ Không tìm thấy vn_stocks_full.txt hoặc file rỗng.\nChạy buildlist trước.'
            )
            return

        start_time = time.time()
        await bot.send_message(
            chat_id=CHAT_ID, parse_mode='HTML',
            text=(
                f"<b>📅 BẮT ĐẦU WEEKLY SCAN</b>\n\n"
                f"Tổng số mã : <b>{total}</b>\n"
                f"Điều kiện  :\n"
                f"  1. Volume tuần &gt; 500,000\n"
                f"  2. RSI(14) cắt lên SMA(RSI,14)\n"
                f"Workers    : 20 threads\n"
                f"Rate limit : 300 req/phút\n\n"
                f"Cập nhật mỗi {WEEKLY_PROGRESS} mã...\n"
                f"🕐 {now_vn()}"
            )
        )

        results, done_cnt = [], [0]
        lock = threading.Lock()
        progress_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

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
                    msg = (
                        f"📊 <b>TIẾN TRÌNH WEEKLY SCAN</b>\n\n"
                        f"Đã xong : {n}/{total} ({n/total*100:.1f}%)\n"
                        f"✅ Tìm thấy: {len(results)} mã\n\n"
                        f"⏱ {elapsed:.0f}s | Còn ~{remain:.0f}s\n"
                        f"🚀 {speed:.0f} mã/phút"
                    )
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

        await asyncio.gather(
            loop.run_in_executor(None, run_and_signal),
            report_progress()
        )

        total_elapsed = time.time() - start_time
        results.sort(key=lambda x: x['symbol'])

        logging.info('─' * 50)
        logging.info('WEEKLY SCAN SUMMARY: %d/%d tín hiệu | %.0fs', len(results), total, total_elapsed)
        logging.info('─' * 50)

        if results:
            header = (
                f"<b>📊 KẾT QUẢ WEEKLY SCAN</b>\n\n"
                f"✅ Thỏa điều kiện: <b>{len(results)}/{total}</b> mã\n"
                f"⏱ {total_elapsed:.0f}s | {total/total_elapsed*60:.0f} mã/phút\n"
                f"🕐 {now_vn()}\n\n"
            )

            # Lấy P/E + cổ tức cho riêng các mã đã lọt điều kiện (ít mã → ít request)
            fund_start = time.time()
            fund_data = {}

            def fetch_fund(r):
                fund_data[r['symbol']] = get_fundamentals(r['symbol'])

            run_pool_sync(fetch_fund, results, max_workers=10)
            logging.info('[fund] Lấy P/E + cổ tức cho %d mã: %.0fs', len(results), time.time() - fund_start)

            lines = [format_result_line(i, r, fund_data.get(r['symbol'], {}))
                     for i, r in enumerate(results, start=1)]
            body = "\n".join(lines)
            full_msg = header + "<pre>" + body + "</pre>"

            # Telegram giới hạn 4096 ký tự/tin nhắn — nếu vượt thì chia nhỏ, không dùng file
            TELEGRAM_LIMIT = 4096
            if len(full_msg) <= TELEGRAM_LIMIT:
                await bot.send_message(chat_id=CHAT_ID, text=full_msg, parse_mode='HTML')
            else:
                await bot.send_message(chat_id=CHAT_ID, text=header, parse_mode='HTML')
                chunk_lines, chunk_len = [], 0
                budget = TELEGRAM_LIMIT - len("<pre></pre>") - 10
                for line in lines:
                    if chunk_len + len(line) + 1 > budget:
                        await bot.send_message(
                            chat_id=CHAT_ID,
                            text="<pre>" + "\n".join(chunk_lines) + "</pre>",
                            parse_mode='HTML'
                        )
                        chunk_lines, chunk_len = [], 0
                    chunk_lines.append(line)
                    chunk_len += len(line) + 1
                if chunk_lines:
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text="<pre>" + "\n".join(chunk_lines) + "</pre>",
                        parse_mode='HTML'
                    )
        else:
            await bot.send_message(
                chat_id=CHAT_ID, parse_mode='HTML',
                text=(
                    f"😔 Không tìm thấy mã nào thỏa điều kiện.\n"
                    f"Tổng quét: {total}\n"
                    f"⏱ {total_elapsed:.0f}s\n"
                    f"🕐 {now_vn()}"
                )
            )

        await bot.send_message(
            chat_id=CHAT_ID,
            text=f"✅ Weekly scan hoàn tất. Thoát.\n🕐 {now_vn()}"
        )

if __name__ == '__main__':
    asyncio.run(main())
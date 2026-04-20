# ============================================================
# BACKTEST BOT v5
# Telegram Commands:
#   [MÃ]             : backtest 1 mã
#   /scanall         : backtest toàn bộ
#   /config          : xem tham số hiện tại
#   /set vol [số]    : volume % MA20        (10-200, mặc định 120)
#   /set trend [số]  : số phiên xu hướng    (0-10,  mặc định 1)
#   /set stop [số]   : trailing stop %      (1-50,  mặc định 10)
#   /year [năm]      : năm bắt đầu backtest (2020-2026, mặc định 2023)
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
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

VN_TZ = timezone(timedelta(hours=7))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

CONFIG = {'vol_pct': 120, 'trend_n': 1, 'stop_pct': 10, 'year': 2023}

now_vn = lambda: datetime.now(VN_TZ).strftime('%Y-%m-%d %H:%M')

# ============================================================
# RATE LIMITER - Token Bucket 150 req/phút
# ============================================================
class RateLimiter:
    def __init__(self, max_calls=150, period=60.0):
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
            raw = [line.strip() for line in f if line.strip()]
        symbols = [s for s in raw if 2 <= len(s) <= 5 and s.isalnum()]
        exclude = {'E1VFVN30', 'FUEKIVFS', 'FUEMAV30', 'FUEMAVND',
                   'FUESSV30', 'FUESSVFL', 'FUETCC50', 'FUEVFVND', 'FUEVN100'}
        return [s for s in dict.fromkeys(symbols) if s not in exclude]
    except:
        return []

# ============================================================
# LẤY DỮ LIỆU - Với retry thông minh
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

    df.index = df.index.normalize()

    rename_map = {'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
    df = df.rename(columns=rename_map).sort_index()
    req_cols = ['Close', 'Volume']
    if not all(c in df.columns for c in req_cols):
        return None
    return df.dropna(subset=req_cols) if not df.empty else None

def get_data(symbol, start_date='2022-01-01'):
    last_errors = []
    for source in ('KBS', 'MSN', 'VCI'):
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
    logging.warning('[get_data] %s failed: %s', symbol, ' | '.join(last_errors))
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

def check_buy_signal(df_w, i, vol_pct, trend_n):
    if i < max(1, trend_n):
        return False
    row, prev = df_w.iloc[i], df_w.iloc[i - 1]
    req = ['Volume', 'ma20_vol', 'rsi', 'sma_rsi']
    if any(pd.isna(row[c]) for c in req) or pd.isna(prev['rsi']) or pd.isna(prev['sma_rsi']):
        return False
    dk1 = row['Volume'] > (vol_pct / 100) * row['ma20_vol']
    dk2 = prev['rsi'] <= prev['sma_rsi'] and row['rsi'] > row['sma_rsi']
    dk3 = all(not (pd.isna(df_w.iloc[i-k]['sma_rsi']) or pd.isna(df_w.iloc[i-k-1]['sma_rsi']) or
                   df_w.iloc[i-k]['sma_rsi'] < df_w.iloc[i-k-1]['sma_rsi'])
              for k in range(trend_n) if i - k - 1 >= 0)
    return dk1 and dk2 and dk3

# ============================================================
# BACKTEST 1 MÃ
# ============================================================
def run_backtest(symbol, initial_capital=50_000_000, vol_pct=None, trend_n=None, stop_pct=None):
    vol_pct  = vol_pct  or CONFIG['vol_pct']
    trend_n  = trend_n  or CONFIG['trend_n']
    stop_pct = stop_pct or CONFIG['stop_pct']
    stop_mult = 1 - stop_pct / 100

    daily, weekly = get_data(symbol)
    if daily is None:
        err = ' | '.join(weekly) if isinstance(weekly, list) else 'unknown'
        return {'error': f'Không lấy được dữ liệu cho mã {symbol} | {err}'}

    df_w = calc_indicators(weekly)
    bt_year = CONFIG['year']
    bt_start = f'{bt_year}-01-01'
    df_w_bt = df_w[df_w.index >= bt_start]
    if df_w_bt.empty:
        return {'error': f'Khong co du lieu tu nam {bt_year}'}

    daily_bt = daily[daily.index >= bt_start].copy()
    daily_list = list(daily_bt.iterrows())

    capital, trades, position, day_idx = initial_capital, [], None, 0

    def do_sell(buy_date, buy_price, sell_date, sell_price, peak, von_vao):
        pct = (sell_price - buy_price) / buy_price * 100
        cap = von_vao * (1 + pct / 100)
        return {
            'stt': len(trades) + 1, 'loai': 'Ban',
            'ngay_mua': buy_date.strftime('%Y-%m-%d'), 'gia_mua': round(buy_price, 2),
            'ngay_ban': sell_date.strftime('%Y-%m-%d'), 'gia_ban': round(sell_price, 2),
            'gia_dinh': round(peak, 2), 'gia_stop': round(peak * stop_mult, 2),
            'von_dau': round(von_vao, 0), 'gia_tri': round(cap, 0),
            'pct': round(pct, 2), 'lai_lo': round(cap - von_vao, 0),
            'von_sau': round(cap, 0), 'dang_giu': False
        }, cap

    for wi, week_end in enumerate(df_w_bt.index.tolist()):
        global_wi = df_w.index.get_loc(week_end)

        if position is not None:
            sold = False
            while day_idx < len(daily_list):
                day_ts, day_row = daily_list[day_idx]
                if day_ts.normalize() > week_end.normalize():
                    break
                if day_ts.normalize() < position['stop_start']:
                    day_idx += 1
                    continue
                stop_price = position['peak'] * stop_mult
                if day_row['Low'] <= stop_price:
                    t, capital = do_sell(position['buy_date'], position['buy_price'],
                                        day_ts, stop_price, position['peak'], position['cost'])
                    trades.append(t)
                    position, sold = None, True
                    day_idx += 1
                    break
                if day_row['High'] > position['peak']:
                    position['peak'] = day_row['High']
                day_idx += 1

            if sold and check_buy_signal(df_w, global_wi, vol_pct, trend_n):
                bp = df_w_bt.iloc[wi]['Close']
                position = {'buy_date': week_end, 'buy_price': bp, 'shares': capital/bp,
                           'cost': capital, 'peak': bp,
                           'stop_start': (week_end + pd.Timedelta(days=3)).normalize()}
            continue

        while day_idx < len(daily_list) and daily_list[day_idx][0].normalize() <= week_end.normalize():
            day_idx += 1
        if check_buy_signal(df_w, global_wi, vol_pct, trend_n):
            bp = df_w_bt.iloc[wi]['Close']
            position = {'buy_date': week_end, 'buy_price': bp, 'shares': capital/bp,
                       'cost': capital, 'peak': bp,
                       'stop_start': (week_end + pd.Timedelta(days=3)).normalize()}

    if position is not None:
        while day_idx < len(daily_list):
            day_ts, day_row = daily_list[day_idx]
            if day_ts.normalize() < position['stop_start']:
                day_idx += 1
                continue
            stop_price = position['peak'] * stop_mult
            if day_row['Low'] <= stop_price:
                t, capital = do_sell(position['buy_date'], position['buy_price'],
                                    day_ts, stop_price, position['peak'], position['cost'])
                trades.append(t)
                position = None
                break
            if day_row['High'] > position['peak']:
                position['peak'] = day_row['High']
            day_idx += 1

        if position is not None:
            last_ts, last_row = daily_list[-1]
            lc = last_row['Close']
            pct = (lc - position['buy_price']) / position['buy_price'] * 100
            current = position['cost'] * (1 + pct / 100)
            capital = current
            trades.append({
                'stt': len(trades)+1, 'loai': 'Dang giu',
                'ngay_mua': position['buy_date'].strftime('%Y-%m-%d'),
                'gia_mua': round(position['buy_price'], 2),
                'ngay_ban': last_ts.strftime('%Y-%m-%d'),
                'gia_ban': round(lc, 2),
                'gia_dinh': round(position['peak'], 2),
                'gia_stop': round(position['peak'] * stop_mult, 2),
                'von_dau': round(position['cost'], 0),
                'gia_tri': round(current, 0),
                'pct': round(pct, 2),
                'lai_lo': round(current - position['cost'], 0),
                'von_sau': round(current, 0),
                'dang_giu': True
            })

    return {
        'symbol': symbol.upper(), 'von_ban_dau': initial_capital,
        'von_cuoi': round(capital, 0), 'lai_lo': round(capital - initial_capital, 0),
        'pct': round((capital / initial_capital - 1) * 100, 2),
        'so_gd': len(trades), 'trades': trades,
        'vol_pct': vol_pct, 'trend_n': trend_n, 'stop_pct': stop_pct,
        'year': CONFIG['year']
    }

# ============================================================
# ĐỊNH DẠNG KẾT QUẢ
# ============================================================
def format_result(r):
    if 'error' in r:
        return [f"Loi: {r['error']}"]

    msgs = [(
        f"<b>BACKTEST {r['symbol']}</b>\n"
        f"Nam: {r.get('year', 2023)} | Vol>{r['vol_pct']}% | Trend {r['trend_n']}p | Stop {r['stop_pct']}%\n"
        f"Von ban dau : {r['von_ban_dau']:,.0f}d\n"
        f"Von cuoi    : {r['von_cuoi']:,.0f}d\n"
        f"Loi nhuan   : {r['lai_lo']:+,.0f}d ({r['pct']:+.2f}%)\n"
        f"So giao dich: {r['so_gd']}"
    )]

    chunk = []
    for t in r['trades']:
        status = 'DANG GIU' if t['dang_giu'] else 'BAN'
        label  = 'Hien tai ' if t['dang_giu'] else 'Ban      '
        chunk.append(
            f"<b>#{t['stt']} {status}</b>\n"
            f"  Mua      : {t['ngay_mua']} @ {t['gia_mua']:,}d\n"
            f"  {label}: {t['ngay_ban']} @ {t['gia_ban']:,}d\n"
            f"  Dinh/Stop: {t['gia_dinh']:,}d / {t['gia_stop']:,}d\n"
            f"  Von vao  : {t['von_dau']:,.0f}d\n"
            f"  Von sau  : {t['von_sau']:,.0f}d\n"
            f"  Lai/Lo   : {t['lai_lo']:+,.0f}d ({t['pct']:+.2f}%)"
        )
        if len(chunk) == 4:
            msgs.append('\n\n'.join(chunk))
            chunk = []
    if chunk:
        msgs.append('\n\n'.join(chunk))
    return msgs

# ============================================================
# HELPER - Pool đồng bộ
# ============================================================
def run_pool_sync(fn, symbols, max_workers=20):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, sym): sym for sym in symbols}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error('[pool] %s', e)

# ============================================================
# PROGRESS REPORTER
# ============================================================
async def report_progress(chat_id, context, queue, task_name):
    while True:
        msg = await queue.get()
        if msg is None:
            break
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
        except Exception as e:
            logging.warning('[%s_progress] %s', task_name, e)

# ============================================================
# HANDLERS
# ============================================================
async def handle_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"<b>Tham so hien tai:</b>\n"
        f"Volume : > {CONFIG['vol_pct']}% MA20  (10-200)\n"
        f"Trend  : {CONFIG['trend_n']} phien        (0-10)\n"
        f"Stop   : {CONFIG['stop_pct']}%            (1-50)\n"
        f"Nam BT : {CONFIG['year']}           (2020-2026)\n\n"
        f"Thay doi:\n"
        f"  /set vol [so]   -> % volume\n"
        f"  /set trend [so] -> so phien xu huong\n"
        f"  /set stop [so]  -> % trailing stop\n"
        f"  /year [nam]     -> nam bat dau backtest",
        parse_mode='HTML'
    )

async def handle_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 2:
        await update.message.reply_text(
            'Cu phap: /set [key] [gia tri]\n'
            '  /set vol 150   -> Volume > 150% MA20\n'
            '  /set trend 3   -> SMA tang trong 3 phien (0=tat dieu kien trend)\n'
            '  /set stop 15   -> Trailing stop 15%'
        )
        return

    key = args[0].lower()
    try:
        val = float(args[1])
    except ValueError:
        await update.message.reply_text('Gia tri phai la so.')
        return

    ranges = {'vol': (10, 200), 'trend': (0, 10), 'stop': (1, 50)}
    if key not in ranges:
        await update.message.reply_text('Key khong hop le. Dung: vol, trend, stop')
        return

    if not (ranges[key][0] <= val <= ranges[key][1]):
        await update.message.reply_text(
            f'{key.capitalize()} phai tu {ranges[key][0]} den {ranges[key][1]}.'
        )
        return

    CONFIG[f'{key}_pct' if key in ['vol', 'stop'] else f'{key}_n'] = int(val) if key != 'stop' else val
    msgs = {
        'vol':   f'Da cap nhat: Volume > {int(val)}% MA20',
        'trend': f'Da cap nhat: Trend SMA {int(val)} phien',
        'stop':  f'Da cap nhat: Trailing stop {val}%'
    }
    await update.message.reply_text(msgs[key])

async def handle_year(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    current_year = datetime.now(VN_TZ).year

    if len(args) != 1:
        await update.message.reply_text(
            f"Cu phap: /year [nam]\n"
            f"  Vi du: /year 2023\n"
            f"  Pham vi hop le: 2020 - {current_year}\n"
            f"  Nam hien tai  : {CONFIG['year']}"
        )
        return

    try:
        year = int(args[0])
    except ValueError:
        await update.message.reply_text('Nam phai la so nguyen. Vi du: /year 2024')
        return

    if not (2020 <= year <= current_year):
        await update.message.reply_text(
            f'Nam phai tu 2020 den {current_year}.\n'
            f'Vi du: /year 2023'
        )
        return

    CONFIG['year'] = year
    await update.message.reply_text(
        f"Da cap nhat: Backtest tu nam <b>{year}</b>\n"
        f"Ap dung cho lenh tiep theo.",
        parse_mode='HTML'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().upper()
    if not (2 <= len(text) <= 5 and text.isalnum()):
        await update.message.reply_text(
            'Nhap ma co phieu (VD: VCB)\n/scanall /config /set'
        )
        return

    await update.message.reply_text(
        f"<b>Dang chay backtest {text}...</b>\n"
        f"Nam: {CONFIG['year']} | Vol>{CONFIG['vol_pct']}% | Trend {CONFIG['trend_n']}p | Stop {CONFIG['stop_pct']}%",
        parse_mode='HTML'
    )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: run_backtest(text))

    for msg in format_result(result):
        await update.message.reply_text(msg, parse_mode='HTML')

# ---------- /scanall ----------
PROGRESS_INTERVAL = 50

async def handle_scanall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = get_all_symbols()
    total = len(symbols)
    chat_id = update.effective_chat.id

    if total == 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text='Khong tim thay file vn_stocks_full.txt hoac file rong.\nChay /buildlist truoc.'
        )
        return

    vol_pct, trend_n, stop_pct = CONFIG['vol_pct'], CONFIG['trend_n'], CONFIG['stop_pct']
    start_time = time.time()

    await context.bot.send_message(
        chat_id=chat_id, parse_mode='HTML',
        text=(
            f"<b>BAT DAU SCAN TOAN BO (BACKTEST)</b>\n"
            f"Tong so ma : <b>{total}</b>\n"
            f"Nam BT     : {CONFIG['year']}\n"
            f"Vol>{vol_pct}% | Trend {trend_n}p | Stop {stop_pct}%\n"
            f"Workers    : 20 threads\n"
            f"Rate limit : 150 req/phut (Bronze)\n"
            f"Cap nhat moi {PROGRESS_INTERVAL} ma..."
        )
    )

    results, errors, done_cnt = [], [], [0]
    lock = threading.Lock()
    loop = asyncio.get_event_loop()
    progress_queue = asyncio.Queue()

    def backtest_one(sym):
        r = run_backtest(sym, vol_pct=vol_pct, trend_n=trend_n, stop_pct=stop_pct)
        with lock:
            done_cnt[0] += 1
            n = done_cnt[0]
            if 'error' not in r:
                results.append({'symbol': sym, 'so_gd': r['so_gd'], 'pct': r['pct'], 'lai_lo': r['lai_lo']})
            else:
                errors.append(sym)

            if n % PROGRESS_INTERVAL == 0 or n == total:
                elapsed = time.time() - start_time
                remaining = (elapsed / n) * (total - n) if n > 0 else 0
                speed = n / elapsed * 60 if elapsed > 0 else 0
                msg = (
                    f"<b>TIEN TRINH BACKTEST</b>\n"
                    f"Da xong : {n}/{total} ({n/total*100:.1f}%)\n"
                    f"OK      : {len(results)}\n"
                    f"Loi     : {len(errors)}\n"
                    f"Da chay  : {elapsed:.0f}s\n"
                    f"Con lai  : ~{remaining:.0f}s\n"
                    f"Toc do   : {speed:.0f} ma/phut"
                )
                asyncio.run_coroutine_threadsafe(progress_queue.put(msg), loop)

            if n == total:
                asyncio.run_coroutine_threadsafe(progress_queue.put(None), loop)

    await asyncio.gather(
        loop.run_in_executor(None, lambda: run_pool_sync(backtest_one, symbols)),
        report_progress(chat_id, context, progress_queue, 'backtest')
    )

    if not results:
        await context.bot.send_message(chat_id=chat_id, text='Khong co du lieu.')
        return

    df_r = pd.DataFrame(results)
    df_gd = df_r[df_r['so_gd'] > 0]
    n_gd = len(df_gd)

    if n_gd == 0:
        await context.bot.send_message(chat_id=chat_id, text='Khong co ma nao co giao dich.')
        return

    n_loi = len(df_gd[df_gd['pct'] > 0])
    n_hoa = len(df_gd[df_gd['pct'] == 0])
    n_lo  = len(df_gd[df_gd['pct'] < 0])
    tong_ll  = df_gd['lai_lo'].sum()
    tb_pct   = df_gd['pct'].mean()
    tong_loi = df_gd.loc[df_gd['lai_lo'] > 0, 'lai_lo'].sum()
    tong_lo_v = df_gd.loc[df_gd['lai_lo'] < 0, 'lai_lo'].sum()
    pf_str = f"{tong_loi / abs(tong_lo_v):.2f}" if tong_lo_v < 0 else 'N/A (khong co ma lo)'
    total_elapsed = time.time() - start_time

    await context.bot.send_message(
        chat_id=chat_id, parse_mode='HTML',
        text=(
            f"<b>KET QUA SCAN TOAN BO (BACKTEST)</b>\n"
            f"Nam BT: {CONFIG['year']} | Vol>{vol_pct}% | Trend {trend_n}p | Stop {stop_pct}%\n"
            f"Tong ma test : {len(df_r)}\n"
            f"Co GD        : {n_gd}\n"
            f"Khong co GD  : {len(df_r[df_r['so_gd'] == 0])}\n"
            f"Loi DL       : {len(errors)}\n"
            f"Trong {n_gd} ma co GD:\n"
            f"  Loi: {n_loi} ({round(n_loi/n_gd*100, 1)}%)\n"
            f"  Hoa: {n_hoa} ({round(n_hoa/n_gd*100, 1)}%)\n"
            f"  Lo : {n_lo} ({round(n_lo/n_gd*100, 1)}%)\n"
            f"Tong lai/lo  : {tong_ll:+,.0f}d\n"
            f"TB/ma        : {tong_ll/n_gd:+,.0f}d ({tb_pct:+.2f}%)\n"
            f"Profit Factor: {pf_str}\n"
            f"(Moi ma von 50tr)\n"
            f"Tong thoi gian: {total_elapsed:.0f}s\n"
            f"Toc do TB     : {len(df_r)/total_elapsed*60:.0f} ma/phut"
        )
    )

    top_loi = df_gd.nlargest(5, 'pct')[['symbol', 'pct', 'lai_lo', 'so_gd']]
    top_lo  = df_gd.nsmallest(5, 'pct')[['symbol', 'pct', 'lai_lo', 'so_gd']]

    msg_loi = '<b>TOP 5 LOI:</b>\n'
    for _, row in top_loi.iterrows():
        msg_loi += f"{row['symbol']}: {row['pct']:+.2f}% | {row['lai_lo']:+,.0f}d | {int(row['so_gd'])} GD\n"
    await context.bot.send_message(chat_id=chat_id, text=msg_loi, parse_mode='HTML')

    msg_lo = '<b>TOP 5 LO:</b>\n'
    for _, row in top_lo.iterrows():
        msg_lo += f"{row['symbol']}: {row['pct']:+.2f}% | {row['lai_lo']:+,.0f}d | {int(row['so_gd'])} GD\n"
    await context.bot.send_message(chat_id=chat_id, text=msg_lo, parse_mode='HTML')

    csv_name = f"ket_qua_scanall_{datetime.now(VN_TZ).strftime('%Y%m%d_%H%M')}.csv"
    df_r.sort_values('pct', ascending=False).to_csv(csv_name, index=False, encoding='utf-8-sig')
    with open(csv_name, 'rb') as f:
        await context.bot.send_document(
            chat_id=chat_id,
            document=f,
            filename=csv_name,
            caption=f"Ket qua backtest {len(df_r)} ma"
        )

# ============================================================
# INIT & MAIN
# ============================================================
async def post_init(app):
    await app.bot.send_message(
        chat_id=CHAT_ID, parse_mode='HTML',
        text=(
            f"<b>BOT SAN SANG!</b>\n"
            f"<b>Lenh:</b>\n"
            f"  [MA]          : backtest 1 ma\n"
            f"  /scanall      : backtest toan bo\n"
            f"  /config       : xem tham so hien tai\n\n"
            f"<b>Chinh tham so:</b>\n"
            f"  /set vol [10-200]  : % volume\n"
            f"  /set trend [1-10]  : so phien xu huong\n"
            f"  /set stop [1-50]   : % trailing stop\n"
            f"  /year [2020-2026]  : nam bat dau backtest\n\n"
            f"Mac dinh: Vol>{CONFIG['vol_pct']}% | Trend {CONFIG['trend_n']}p | Stop {CONFIG['stop_pct']}% | Nam {CONFIG['year']}\n"
            f"Von backtest: 50tr/ma\n"
            f"Rate limit: 150 req/phut | 20 workers"
        )
    )

def main():
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler('scanall', handle_scanall))
    app.add_handler(CommandHandler('config',  handle_config))
    app.add_handler(CommandHandler('set',     handle_set))
    app.add_handler(CommandHandler('year',    handle_year))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print('Bot dang chay...')
    app.run_polling()

if __name__ == '__main__':
    main()

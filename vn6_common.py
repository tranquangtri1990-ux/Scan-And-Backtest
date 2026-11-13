# ============================================================
# vn6_common.py - Lớp dữ liệu dùng chung cho 3 bot (vnstock thế hệ 6)
#
# Khác biệt chính so với vnstock 5:
#   - Không còn Vnstock/Quote/Listing, không còn nguồn VCI/KBS/MSN.
#   - Chỉ còn Open API chính thức của CTCK (dnse, ssi, tcbs, fhsc, fmp).
#     Bạn có TCBS_API_KEY -> dùng nguồn "tcbs".
#   - TCBS cần Smart OTP để mở phiên (8-24h). OTP KHÔNG được lưu xuống máy,
#     nên mỗi lần chạy trên GitHub Actions phải mở phiên bằng OTP mới.
#     -> Bot gửi tin nhắn Telegram, bạn trả lời:  /otp 123456
#   - Không cần vnstock.login(VNSTOCK_API_KEY) cho Market/Reference.
#   - Gọi dữ liệu: Market("tcbs").equity(sym).ohlcv(start, end, interval)
# ============================================================

import os
import re
import time
import asyncio
import logging
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd

VN_TZ = timezone(timedelta(hours=7))

# Thứ tự nguồn dữ liệu (mặc định chỉ tcbs). Sau này có thêm khóa dnse/ssi
# thì đặt DATA_SOURCES="tcbs,dnse" để có nguồn dự phòng.
DATA_SOURCES = [s.strip().lower() for s in os.environ.get("DATA_SOURCES", "tcbs").split(",") if s.strip()]

# Bỏ nến cuối nếu nguồn báo là nến chưa đóng (đang trong phiên)
DROP_PARTIAL_BAR = os.environ.get("DROP_PARTIAL_BAR", "1") == "1"

EXCLUDE = {
    "E1VFVN30", "FUEKIVFS", "FUEMAV30", "FUEMAVND",
    "FUESSV30", "FUESSVFL", "FUETCC50", "FUEVFVND", "FUEVN100",
}


# ============================================================
# RATE LIMITER
# TCBS khuyến nghị < 10 request/giây; mỗi lần ohlcv() có thể gồm nhiều
# request nội bộ nên để mặc định 240 lần gọi/phút (chỉnh bằng RATE_PER_MIN).
# ============================================================
class RateLimiter:
    def __init__(self, max_calls=240, period=60.0):
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


rate_limiter = RateLimiter(int(os.environ.get("RATE_PER_MIN", "240")))


# ============================================================
# LỖI & TRẠNG THÁI PHIÊN
# ============================================================
class SessionExpired(Exception):
    """Phiên TCBS chưa mở / đã hết hạn - cần OTP mới."""


SESSION_DEAD = threading.Event()


def _name(e):
    return type(e).__name__


def is_rate_limited(e):
    s = str(e).lower()
    return "ratelimit" in _name(e).lower() or any(
        k in s for k in ("rate limit", "rate_limit", "429", "too many"))


def is_otp_required(e):
    s = str(e).lower()
    return _name(e) in ("OtpRequiredError", "LoginRequiredError") or "otp" in s or "login_required" in s


def is_credential_error(e):
    return _name(e) in ("CredentialError", "AuthError", "NoKeyringError")


# ============================================================
# CHUẨN HOÁ BẢNG NẾN
# ============================================================
def _normalize(df):
    """Trả về DataFrame index=time (tz Asia/Ho_Chi_Minh), cột Open/High/Low/Close/Volume.
    Giá tính bằng NGHÌN ĐỒNG (giữ như thế hệ 5)."""
    if df is None or getattr(df, "empty", True):
        return None
    partial = bool(getattr(df, "attrs", {}).get("last_bar_partial"))
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "volume": "Volume"}).sort_index()
    if not all(c in df.columns for c in ("Close", "Volume")):
        return None
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close", "Volume"])
    if DROP_PARTIAL_BAR and partial and len(df) > 1:
        df = df.iloc[:-1]
    return df if not df.empty else None


def fetch_daily(symbol, start, source="tcbs", end=None):
    """Nến ngày 1 mã từ 1 nguồn. Ném SessionExpired nếu cần OTP."""
    if SESSION_DEAD.is_set() and source == "tcbs":
        raise SessionExpired("phiên TCBS đã hết hạn")
    from vnstock import Market
    end = end or datetime.now(VN_TZ).strftime("%Y-%m-%d")
    rate_limiter.acquire()
    try:
        raw = Market(source).equity(symbol).ohlcv(start=start, end=end, interval="1D")
    except Exception as e:
        if is_otp_required(e):
            SESSION_DEAD.set()
            raise SessionExpired(str(e)[:200]) from e
        raise
    return _normalize(raw)


def to_weekly(daily):
    return daily.resample("W-FRI").agg({"Close": "last", "Volume": "sum"}).dropna()


def get_data(symbol, start_date="2022-01-01"):
    """Trả về (daily, weekly) hoặc (None, [danh sách lỗi])."""
    errors = []
    for source in DATA_SOURCES:
        for attempt in range(3):
            try:
                df = fetch_daily(symbol, start_date, source)
                if df is not None:
                    return df, to_weekly(df)
                errors.append(f"{source}:empty")
                break
            except SessionExpired:
                raise
            except Exception as e:
                err = str(e)
                if is_rate_limited(e):
                    logging.warning("[get_data] %s/%s rate limit, chờ %ds", symbol, source, 15 * (attempt + 1))
                    time.sleep(15 * (attempt + 1))
                    continue
                if is_credential_error(e):
                    errors.append(f"{source}:credential:{err[:80]}")
                    break
                logging.warning("[get_data] %s/%s: %s", symbol, source, err[:120])
                errors.append(f"{source}:{err[:120]}")
                break
    return None, errors


# ============================================================
# DANH SÁCH MÃ  (vnstock 6: Reference chỉ có dnse/ssi -> với TCBS dùng
# bảng room ngoại theo sàn để lấy toàn bộ mã đang giao dịch)
# ============================================================
def clean_symbols(raw):
    syms = [str(s).strip().strip("[]").upper() for s in raw]
    syms = [s for s in syms if re.fullmatch(r"[A-Z][A-Z0-9]{1,4}", s)]
    syms = [s for s in dict.fromkeys(syms) if s not in EXCLUDE and not s.startswith("FUE")]
    return syms


def list_symbols_tcbs():
    from vnstock import Market
    out = []
    for ex in ("HOSE", "HNX", "UPCOM"):
        for attempt in range(3):
            try:
                rate_limiter.acquire()
                df = Market("tcbs").equity.foreign_room_board(exchange=ex)
                if df is not None and not df.empty:
                    cols = [c.lower() for c in df.columns]
                    df.columns = cols
                    out += df["symbol"].astype(str).tolist()
                    logging.info("[listing] %s: %d mã", ex, len(df))
                break
            except Exception as e:
                if is_otp_required(e):
                    SESSION_DEAD.set()
                    raise SessionExpired(str(e)[:200]) from e
                logging.warning("[listing] %s lần %d: %s", ex, attempt + 1, str(e)[:120])
                time.sleep(5 * (attempt + 1))
    return clean_symbols(out)


# ============================================================
# PHIÊN TCBS (Smart OTP)
# ============================================================
def open_session(otp, source="tcbs"):
    """Mở phiên trong tiến trình hiện tại. Cần VNSTOCK_OAPI_SESSION_BACKEND=file
    trên máy không có keyring (GitHub Actions)."""
    import vnstock_oapi as oapi
    oapi.source(source).authenticate(otp)
    SESSION_DEAD.clear()


def check_session(source="tcbs"):
    """(True,'') nếu gọi được dữ liệu; (False,'otp') nếu cần OTP; (False,lỗi) nếu lỗi khác."""
    try:
        start = (datetime.now(VN_TZ) - timedelta(days=10)).strftime("%Y-%m-%d")
        df = fetch_daily("FPT", start, source)
        if df is None:
            return False, "FPT trả về rỗng (kiểm tra khóa/giờ phục vụ)"
        SESSION_DEAD.clear()
        return True, ""
    except SessionExpired:
        return False, "otp"
    except Exception as e:
        if is_otp_required(e):
            return False, "otp"
        return False, f"{_name(e)}: {str(e)[:200]}"


OTP_RE = re.compile(r"/?otp\s+(\d{4,8})", re.I)


async def ensure_tcbs_session(bot, chat_id, wait_min=15):
    """Đảm bảo có phiên TCBS. Nếu chưa: lấy OTP từ env TCBS_OTP hoặc chờ bạn
    nhắn  /otp 123456  trên Telegram (tin nhắn OTP được xoá ngay sau khi đọc)."""
    loop = asyncio.get_running_loop()
    ok, why = await loop.run_in_executor(None, check_session)
    if ok:
        return True
    if why != "otp":
        await bot.send_message(chat_id=chat_id, text=f"❌ Không gọi được dữ liệu TCBS:\n{why}")
        return False

    env_otp = os.environ.get("TCBS_OTP", "").strip()
    if env_otp:
        try:
            await loop.run_in_executor(None, open_session, env_otp)
            ok, why = await loop.run_in_executor(None, check_session)
            if ok:
                return True
        except Exception as e:
            logging.warning("[otp] TCBS_OTP lỗi: %s", e)

    # bỏ qua các tin cũ
    offset = None
    try:
        old = await bot.get_updates(offset=-1, timeout=0)
        if old:
            offset = old[-1].update_id + 1
    except Exception:
        pass

    await bot.send_message(
        chat_id=chat_id,
        text=(f"🔐 Cần Smart OTP TCBS để mở phiên.\n"
              f"Mở app TCBS lấy mã rồi nhắn:  /otp 123456\n"
              f"(chờ tối đa {wait_min} phút; tin nhắn OTP sẽ được xoá)")
    )
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        try:
            ups = await bot.get_updates(offset=offset, timeout=20, read_timeout=30,
                                        allowed_updates=["message"])
        except Exception as e:
            logging.warning("[otp] getUpdates: %s", e)
            await asyncio.sleep(3)
            continue
        for u in ups:
            offset = u.update_id + 1
            m = u.message
            if not m or str(m.chat_id) != str(chat_id) or not m.text:
                continue
            mt = OTP_RE.fullmatch(m.text.strip())
            if not mt:
                continue
            try:
                await m.delete()
            except Exception:
                pass
            try:
                await loop.run_in_executor(None, open_session, mt.group(1))
            except Exception as e:
                await bot.send_message(chat_id=chat_id,
                                       text=f"❌ OTP không mở được phiên: {_name(e)}\nLấy mã mới và gửi lại.")
                continue
            ok, why = await loop.run_in_executor(None, check_session)
            if ok:
                await bot.send_message(chat_id=chat_id, text="✅ Đã mở phiên TCBS.")
                return True
            await bot.send_message(chat_id=chat_id, text=f"❌ Mở phiên xong nhưng gọi dữ liệu lỗi: {why}")
    await bot.send_message(chat_id=chat_id, text="⏱ Hết thời gian chờ OTP. Chạy lại workflow.")
    return False

# ============================================================
# tcbs_common.py - Lớp dữ liệu dùng chung cho 3 bot (thay vn6_common.py)
# KHÔNG dùng vnstock / vnstock-oapi. Gọi thẳng TCBS iFlash OpenAPI.
#
# Tên hàm/biến giữ giống vn6_common để buildlist / weeklyscan / backtest
# chỉ cần đổi dòng import:   import tcbs_common as vc
#
# Endpoint (tài liệu TCBS iFlash Open API v1.0.0, https://openapi.tcbs.com.vn):
#   POST /gaia/v1/oauth2/openapi/token        {apiKey, otp} -> {token}
#   GET  /tartarus/v1/tickerCommons?index=N   mục 5.1  (1=HOSE, 3=HNX, 5=UPCOM)
#   GET  /tartarus/v1/tickerSnaps?index=N     mục 5.3
#   GET  /nyx/v2/stock/bars-long-term         mục 5.13 (resolution D/W/M)
#
# Phiên TCBS cần Smart OTP (không lưu xuống máy) -> mỗi lần chạy trên GitHub
# Actions phải nhắn Telegram:  /otp 123456
#
# Biến môi trường: TCBS_API_KEY, RATE_PER_MIN (mặc định 240), TCBS_OTP (tuỳ chọn),
#   DROP_PARTIAL_BAR (mặc định 1), BARS_CHUNK (mặc định 600), TCBS_BASE_URL (tuỳ chọn)
# ============================================================

import asyncio
import logging
import os
import re
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone

import pandas as pd
import requests

BASE = os.environ.get("TCBS_BASE_URL", "https://openapi.tcbs.com.vn").rstrip("/")
VN_TZ = timezone(timedelta(hours=7))

DATA_SOURCES = ["tcbs"]     # giữ tên cũ để code gọi không phải đổi

# Bỏ nến ngày cuối nếu là phiên hôm nay và chưa đóng cửa (trước 15:10)
DROP_PARTIAL_BAR = os.environ.get("DROP_PARTIAL_BAR", "1") == "1"

BOARDS = {"HOSE": 1, "HNX": 3, "UPCOM": 5}   # sau 28/12/2026 HNX sẽ trống, vẫn để đây
LIST_ENDPOINTS = ("/tartarus/v1/tickerCommons", "/tartarus/v1/tickerSnaps")
BARS_ENDPOINT = "/nyx/v2/stock/bars-long-term"
TIMEOUT = 30
BARS_CHUNK = int(os.environ.get("BARS_CHUNK", "600"))   # số nến tối đa mỗi lần gọi
MAX_PAGES = 8                                           # số lần gọi tối đa mỗi mã

EXCLUDE = {
    "E1VFVN30", "FUEKIVFS", "FUEMAV30", "FUEMAVND",
    "FUESSV30", "FUESSVFL", "FUETCC50", "FUEVFVND", "FUEVN100",
}

BOARD_COUNTS = {}           # số mã từng sàn ở lần lấy danh sách gần nhất


# ============================================================
# RATE LIMITER (giữ nguyên bản vn6_common)
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


rate_limiter = RateLimiter(int(float(os.environ.get("RATE_PER_MIN", "240"))))


# ============================================================
# LỖI & TRẠNG THÁI PHIÊN
# ============================================================
class SessionExpired(Exception):
    """Phiên TCBS chưa mở / đã hết hạn - cần OTP mới."""


class Retryable(Exception):
    """Lỗi tạm thời: 429, 5xx, lỗi mạng. Gọi lại sau một lúc."""


class CredentialError(Exception):
    """Thiếu hoặc sai TCBS_API_KEY."""


SESSION_DEAD = threading.Event()
_state = {"token": None, "div": 1000.0}   # div: chia giá về nghìn đồng (1000 nếu API trả đồng)


def _name(e):
    return type(e).__name__


def is_rate_limited(e):
    if isinstance(e, Retryable):
        return True
    s = str(e).lower()
    return "ratelimit" in _name(e).lower() or any(
        k in s for k in ("rate limit", "rate_limit", "429", "too many"))


def is_otp_required(e):
    return isinstance(e, SessionExpired)


def is_credential_error(e):
    return isinstance(e, CredentialError)


# ============================================================
# HTTP
# ============================================================
_local = threading.local()


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


def get_token(api_key, otp):
    """Đổi API Key + Smart OTP lấy JWT token (TCBS giới hạn 10 lần/ngày)."""
    if not api_key:
        raise CredentialError("thiếu TCBS_API_KEY")
    r = requests.post(
        f"{BASE}/gaia/v1/oauth2/openapi/token",
        json={"apiKey": api_key, "otp": otp},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    token = body.get("token") or (body.get("data") or {}).get("token")
    if not token:
        raise RuntimeError(f"không thấy token trong phản hồi: {str(body)[:200]}")
    return token


def _http_get(path, params):
    """GET có xác thực. Trả về JSON; ném SessionExpired / Retryable / RuntimeError."""
    if SESSION_DEAD.is_set():
        raise SessionExpired("phiên TCBS đã hết hạn")
    if not _state["token"]:
        raise SessionExpired("chưa mở phiên TCBS")
    rate_limiter.acquire()
    headers = {"Authorization": f"Bearer {_state['token']}", "Content-Type": "application/json"}
    try:
        r = _session().get(f"{BASE}{path}", params=params, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise Retryable(f"lỗi mạng: {e}") from e
    if r.status_code == 401:
        SESSION_DEAD.set()
        raise SessionExpired("401: token không hợp lệ hoặc hết hạn")
    if r.status_code == 429 or r.status_code >= 500:
        raise Retryable(f"HTTP {r.status_code}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


# ============================================================
# DANH SÁCH MÃ
# ============================================================
def clean_symbols(raw):
    """Làm sạch danh sách mã đọc từ file (giữ nguyên quy tắc vn6_common)."""
    syms = [str(s).strip().strip("[]").upper() for s in raw]
    syms = [s for s in syms if re.fullmatch(r"[A-Z][A-Z0-9]{1,4}", s)]
    return [s for s in dict.fromkeys(syms) if s not in EXCLUDE and not s.startswith("FUE")]


def _stock_symbols(raw):
    """Cổ phiếu VN có mã đúng 3 ký tự, bắt đầu bằng chữ (có thể có số: PC1, L14, G36).
    Loại chứng quyền (8 ký tự), ETF/quỹ, phái sinh."""
    out = {str(s).strip().upper() for s in raw}
    return sorted(s for s in out if re.fullmatch(r"[A-Z][A-Z0-9]{2}", s) and s not in EXCLUDE)


def list_symbols_tcbs():
    """TOÀN BỘ mã cổ phiếu HOSE + HNX + UPCOM: gộp tickerCommons và tickerSnaps theo từng sàn."""
    universe, counts = set(), {}
    for name, idx in BOARDS.items():
        board = set()
        for ep in LIST_ENDPOINTS:
            for attempt in range(3):
                try:
                    body = _http_get(ep, {"index": idx})
                except Retryable:
                    time.sleep(3 * (attempt + 1))
                    continue
                except SessionExpired:
                    raise
                except Exception as e:  # noqa: BLE001
                    logging.warning("[listing] %s?index=%s lỗi: %s", ep, idx, str(e)[:120])
                    break
                board |= {str(i.get("symbol", "")) for i in (body.get("data") or [])}
                break
            else:
                logging.warning("[listing] %s?index=%s lỗi sau 3 lần thử", ep, idx)
        board = set(_stock_symbols(board))
        counts[name] = len(board)
        logging.info("[listing] %s: %d mã", name, len(board))
        universe |= board
    BOARD_COUNTS.clear()
    BOARD_COUNTS.update(counts)
    return sorted(universe)


# ============================================================
# NẾN
# ============================================================
def _fetch_bars(symbol, resolution, count, to=None):
    body = _http_get(BARS_ENDPOINT, {
        "ticker": symbol,
        "type": "stock",
        "resolution": resolution,
        "to": int(to if to is not None else time.time()),
        "countBack": int(count),
    })
    return body.get("data") or []


def _bars_to_df(raw):
    """Danh sách nến TCBS -> DataFrame (index = ngày giao dịch giờ VN, tz-naive, giá nghìn đồng).
    Thêm cột phụ _ts (epoch giây) để phân trang."""
    df = pd.DataFrame(raw)
    if not {"close", "volume", "tradingDate"}.issubset(df.columns):
        return None
    ts = pd.to_datetime(df["tradingDate"], utc=True, errors="coerce")
    div = _state["div"]
    num = lambda col: pd.to_numeric(df[col] if col in df.columns else df["close"], errors="coerce")
    out = pd.DataFrame({
        "Open": num("open") / div,
        "High": num("high") / div,
        "Low": num("low") / div,
        "Close": num("close") / div,
        "Volume": pd.to_numeric(df["volume"], errors="coerce"),
        "_ts": ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).fillna(0),
    })
    out.index = ts.dt.tz_convert(VN_TZ).dt.tz_localize(None).dt.normalize()
    out = out[out.index.notna() & ts.notna().values & out["Close"].notna() & out["Volume"].notna()]
    return out if len(out) else None


def calibrate_price_unit():
    """Giá nến theo tài liệu là đồng (vd 25500); các bot cũ dùng nghìn đồng.
    Kiểm tra bằng mã bluechip: >= 1000 thì là đồng (chia 1000), ngược lại giữ nguyên."""
    for ref in ("VCB", "FPT", "HPG", "VNM"):
        try:
            raw = _fetch_bars(ref, "D", 3)
            close = float(raw[-1]["close"])
        except SessionExpired:
            raise
        except Exception:  # noqa: BLE001
            continue
        _state["div"] = 1000.0 if close >= 1000 else 1.0
        logging.info("Đơn vị giá TCBS: %s (%s close=%s)",
                     "đồng" if _state["div"] == 1000.0 else "nghìn đồng", ref, close)
        return _state["div"]
    _state["div"] = 1000.0
    logging.warning("Không hiệu chỉnh được đơn vị giá, mặc định là đồng (chia 1000)")
    return _state["div"]


def fetch_daily(symbol, start, source="tcbs", end=None):
    """Nến ngày 1 mã từ `start` (YYYY-MM-DD) đến `end` (mặc định: hôm nay).
    DataFrame index = ngày, cột Open/High/Low/Close/Volume, giá NGHÌN ĐỒNG.
    Tự phân trang lùi theo `to` nên không phụ thuộc giới hạn countBack của TCBS.
    Ném SessionExpired nếu cần OTP; trả về None nếu không có dữ liệu."""
    if SESSION_DEAD.is_set():
        raise SessionExpired("phiên TCBS đã hết hạn")
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    if end:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        to_ts = int(datetime(end_dt.year, end_dt.month, end_dt.day, 23, 59, 59, tzinfo=VN_TZ).timestamp())
    else:
        end_dt = datetime.now(VN_TZ).date()
        to_ts = int(time.time())

    frames, earliest = [], None
    for _ in range(MAX_PAGES):
        ref = earliest or end_dt
        need = int((ref - start_dt).days * 5 / 7) + 15
        count = min(max(need, 5), BARS_CHUNK)
        raw = _fetch_bars(symbol, "D", count, to_ts)
        if not raw:
            break
        df = _bars_to_df(raw)
        if df is None:
            break
        frames.append(df)
        new_earliest = df.index.min().date()
        if earliest is not None and new_earliest >= earliest:
            break                                    # không lùi thêm được nữa
        earliest = new_earliest
        if earliest <= start_dt:
            break                                    # đã đủ lịch sử
        # Có thể TCBS giới hạn số nến mỗi lần gọi (< countBack) hoặc mã đã hết lịch sử:
        # gọi tiếp lùi theo `to`; nếu không lùi thêm được thì vòng sau tự dừng.
        to_ts = int(df["_ts"].min()) - 1

    if not frames:
        return None
    out = pd.concat(frames).drop(columns="_ts")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out[(out.index >= pd.Timestamp(start_dt)) & (out.index <= pd.Timestamp(end_dt))]
    if DROP_PARTIAL_BAR and len(out) > 1:
        now = datetime.now(VN_TZ)
        if out.index[-1].date() == now.date() and now.time() < dtime(15, 10):
            out = out.iloc[:-1]
    return out if not out.empty else None


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
# PHIÊN TCBS (Smart OTP)
# ============================================================
def open_session(otp, source="tcbs"):
    """Đổi API Key + OTP lấy token cho tiến trình hiện tại và hiệu chỉnh đơn vị giá."""
    _state["token"] = get_token(os.environ.get("TCBS_API_KEY", ""), otp)
    SESSION_DEAD.clear()
    calibrate_price_unit()


def check_session(source="tcbs"):
    """(True,'') nếu gọi được dữ liệu; (False,'otp') nếu cần OTP; (False,lỗi) nếu lỗi khác."""
    if not _state["token"]:
        return False, "otp"
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
    deadline, tries = time.time() + wait_min * 60, 0
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
                tries += 1
                await bot.send_message(chat_id=chat_id,
                                       text=f"❌ OTP không mở được phiên: {_name(e)}: {str(e)[:150]}\n"
                                            f"Lấy mã mới và gửi lại.")
                if tries >= 3:          # TCBS giới hạn đổi token 10 lần/ngày
                    return False
                continue
            ok, why = await loop.run_in_executor(None, check_session)
            if ok:
                await bot.send_message(chat_id=chat_id, text="✅ Đã mở phiên TCBS.")
                return True
            await bot.send_message(chat_id=chat_id, text=f"❌ Mở phiên xong nhưng gọi dữ liệu lỗi: {why}")
    await bot.send_message(chat_id=chat_id, text="⏱ Hết thời gian chờ OTP. Chạy lại workflow.")
    return False

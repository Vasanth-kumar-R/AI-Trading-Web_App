# app.py — AI Professional Trading Platform
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Dark trading theme */
    .stApp { background-color: #0a0e1a; color: #e0e6f0; }
    .main .block-container { padding: 1rem 2rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0d1526, #1a2340);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #7a9cc4 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4aa !important; font-size: 1.4rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.9rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060b14 0%, #0d1526 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Signal boxes */
    .signal-enter {
        background: linear-gradient(135deg, #003d1f, #005a2b);
        border: 2px solid #00c853; border-radius: 10px;
        padding: 16px; text-align: center; font-size: 1.5rem; font-weight: bold;
        color: #00e676; animation: pulse 1.5s infinite;
    }
    .signal-exit {
        background: linear-gradient(135deg, #3d0000, #5a0000);
        border: 2px solid #f44336; border-radius: 10px;
        padding: 16px; text-align: center; font-size: 1.5rem; font-weight: bold;
        color: #ff5252; animation: pulse 1.5s infinite;
    }
    .signal-hold {
        background: linear-gradient(135deg, #1a2d00, #2a4500);
        border: 2px solid #76ff03; border-radius: 10px;
        padding: 16px; text-align: center; font-size: 1.5rem; font-weight: bold;
        color: #b2ff59;
    }
    .signal-wait {
        background: linear-gradient(135deg, #1a1500, #2a2000);
        border: 2px solid #ffd600; border-radius: 10px;
        padding: 16px; text-align: center; font-size: 1.5rem; font-weight: bold;
        color: #ffea00;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 10px rgba(0,200,83,0.3); }
        50% { box-shadow: 0 0 25px rgba(0,200,83,0.7); }
    }

    /* RSI gauge */
    .rsi-bar { height: 20px; border-radius: 10px; margin: 5px 0; }
    .indicator-card {
        background: #0d1526; border: 1px solid #1e3a5f;
        border-radius: 8px; padding: 12px; margin: 4px 0;
    }

    /* P&L card */
    .pnl-positive { color: #00e676; font-size: 1.8rem; font-weight: bold; }
    .pnl-negative { color: #ff5252; font-size: 1.8rem; font-weight: bold; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: #0d1526; border: 1px solid #1e3a5f;
        border-radius: 6px; color: #7a9cc4; margin: 2px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important; color: #00d4aa !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #0d2340);
        border: 1px solid #00d4aa; color: #00d4aa;
        border-radius: 6px; font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4aa, #00b894);
        color: #000 !important;
    }

    /* Headers */
    h1, h2, h3 { color: #00d4aa !important; }
    .stCaption { color: #7a9cc4; }

    /* Data source status */
    .source-ok  { color: #00e676; font-weight: bold; }
    .source-err { color: #ff5252; font-weight: bold; }

    /* Trade panel */
    .trade-card {
        background: linear-gradient(135deg, #0d1526, #1a2340);
        border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
defaults = {
    "messages": [],
    "position": {"active": False, "type": None, "strike": None,
                 "entry": None, "sl": None, "target": None,
                 "premium": None, "time": None, "qty": 1},
    "query": "",
    "trade_log": [],
    "last_analysis": None,
    "last_fetch_time": None,
    "consensus_price": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# DATA SOURCES
# ============================================================
def get_yfinance_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="1mo", interval="1d")
        intraday = ticker.history(period="5d", interval="5m")
        return {
            "source": "YFinance", "status": "✅", "ok": True,
            "price": info.get("regularMarketPrice"),
            "open": info.get("regularMarketOpen"),
            "high": info.get("regularMarketDayHigh"),
            "low": info.get("regularMarketDayLow"),
            "volume": info.get("regularMarketVolume"),
            "change_pct": round(info.get("regularMarketChangePercent", 0), 2),
            "prev_close": info.get("regularMarketPreviousClose"),
            "hist": hist,
            "intraday": intraday,
        }
    except Exception as e:
        return {"source": "YFinance", "status": "❌", "ok": False, "error": str(e)}


def get_nse_data():
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        data = n.stock_quote_index("NIFTY BANK")
        p = data['priceInfo']
        return {
            "source": "NSE Direct", "status": "✅", "ok": True,
            "price": p['lastPrice'],
            "open": p['open'],
            "high": p['intraDayHighLow']['max'],
            "low": p['intraDayHighLow']['min'],
            "prev_close": p['previousClose'],
            "change_pct": p['pChange'],
        }
    except Exception as e:
        return {"source": "NSE Direct", "status": "❌", "ok": False, "error": str(e)}


def get_yahoo_data(symbol):
    try:
        encoded = symbol.replace("^", "%5E")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = r.json()
        meta = data['chart']['result'][0]['meta']
        quotes = data['chart']['result'][0]['indicators']['quote'][0]
        highs = [h for h in quotes.get('high', []) if h]
        lows  = [l for l in quotes.get('low',  []) if l]
        return {
            "source": "Yahoo API", "status": "✅", "ok": True,
            "price": meta.get("regularMarketPrice"),
            "open": meta.get("chartPreviousClose"),
            "high": round(max(highs), 2) if highs else None,
            "low":  round(min(lows),  2) if lows  else None,
            "volume": meta.get("regularMarketVolume"),
            "change_pct": round(
                ((meta.get("regularMarketPrice", 0) - meta.get("previousClose", 1))
                 / meta.get("previousClose", 1)) * 100, 2),
        }
    except Exception as e:
        return {"source": "Yahoo API", "status": "❌", "ok": False, "error": str(e)}


def fetch_all(symbol):
    s1, s2, s3 = get_yfinance_data(symbol), get_nse_data(), get_yahoo_data(symbol)
    sources = [s1, s2, s3]
    valid = [float(s['price']) for s in sources if s.get('ok') and s.get('price')]
    consensus = round(sum(valid) / len(valid), 2) if valid else None
    diff = round(max(valid) - min(valid), 2) if len(valid) > 1 else 0
    quality = "✅ Consistent" if diff < 50 else f"⚠️ Discrepancy ₹{diff}"
    return sources, consensus, diff, quality, s1  # return yf data separately


# ============================================================
# TECHNICAL INDICATORS (computed locally)
# ============================================================
def compute_indicators(hist_df):
    if hist_df is None or hist_df.empty or len(hist_df) < 20:
        return {}
    df = hist_df.copy()
    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # SMAs
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() if len(df) >= 50 else np.nan
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD']  = df['EMA12'] - df['EMA26']
    df['Signal']= df['MACD'].ewm(span=9).mean()
    # Bollinger Bands
    df['BB_mid']   = df['Close'].rolling(20).mean()
    df['BB_std']   = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    last = df.iloc[-1]
    rsi  = round(last['RSI'], 1) if not np.isnan(last['RSI']) else None
    if rsi:
        if rsi < 30:   rsi_signal, rsi_color = "🟢 OVERSOLD  → CALL bias", "#00e676"
        elif rsi > 70: rsi_signal, rsi_color = "🔴 OVERBOUGHT → PUT bias",  "#ff5252"
        else:          rsi_signal, rsi_color = "🟡 NEUTRAL",                 "#ffd600"
    else:
        rsi_signal, rsi_color = "N/A", "#7a9cc4"

    price = last['Close']
    sma20 = round(last['SMA20'], 2) if not np.isnan(last['SMA20']) else None
    trend = "📈 BULLISH" if sma20 and price > sma20 else "📉 BEARISH"

    macd   = round(last['MACD'],   2) if not np.isnan(last['MACD'])   else None
    signal = round(last['Signal'], 2) if not np.isnan(last['Signal']) else None
    macd_signal = "BUY" if macd and signal and macd > signal else "SELL"

    return {
        "rsi": rsi, "rsi_signal": rsi_signal, "rsi_color": rsi_color,
        "sma20": sma20,
        "sma50": round(last['SMA50'], 2) if not np.isnan(last.get('SMA50', float('nan'))) else None,
        "macd": macd, "macd_signal_line": signal, "macd_signal": macd_signal,
        "bb_upper": round(last['BB_upper'], 2) if not np.isnan(last['BB_upper']) else None,
        "bb_lower": round(last['BB_lower'], 2) if not np.isnan(last['BB_lower']) else None,
        "trend": trend, "df": df,
    }


def compute_market_regime(ind, consensus):
    """Detect sideways / trending / volatile"""
    if not ind or not consensus:
        return "UNKNOWN"
    if ind.get('bb_upper') and ind.get('bb_lower'):
        band_width = ind['bb_upper'] - ind['bb_lower']
        band_pct   = (band_width / consensus) * 100
        if band_pct < 1.5:
            return "SIDEWAYS"
        elif band_pct > 3:
            return "VOLATILE"
        else:
            return "TRENDING"
    return "UNKNOWN"


def get_scalping_timeframe(regime):
    if regime == "SIDEWAYS":
        return "5min / 15min scalping recommended ✂️"
    elif regime == "TRENDING":
        return "Follow trend — 15min / 30min setups"
    elif regime == "VOLATILE":
        return "Caution — wide stops needed, straddle possible"
    return "Assess chart manually"


def smart_strike(consensus, option_type, step=100):
    """ATM/ITM strike selection"""
    atm = round(consensus / step) * step
    if option_type == "CALL":
        itm = atm - step
        otm = atm + step
    else:
        itm = atm + step
        otm = atm - step
    return {"ATM": atm, "ITM": itm, "OTM": otm}


# ============================================================
# P&L TRACKER
# ============================================================
def compute_pnl(pos, current_price):
    if not pos["active"]:
        return None
    entry = pos["entry"]
    premium = pos["premium"]
    qty = pos.get("qty", 1)
    lot_size = 15  # BankNifty lot size
    if pos["type"] == "CALL":
        index_pnl = current_price - entry
    else:
        index_pnl = entry - current_price
    # Rough option premium change (0.5 delta approximation)
    option_pnl = index_pnl * 0.5 * lot_size * qty
    pnl_pct = (index_pnl / entry) * 100
    return {
        "index_pnl": round(index_pnl, 2),
        "option_pnl": round(option_pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "qty": qty,
        "lot_size": lot_size,
    }


# ============================================================
# MACRO NEWS AGENT
# ============================================================
@st.cache_resource
def get_news_agent():
    return Agent(
        name="News Agent",
        model=Groq(id="llama-3.1-8b-instant"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Search for latest news about: RBI policy, Fed news, Indian budget, global market cues.",
            "Return a concise 5-point bullet summary.",
            "Focus on news that affects Indian banking sector and BankNifty.",
        ],
        markdown=True,
    )


@st.cache_resource
def get_reasoning_agent(model):
    return Agent(
        name="Trading Analyst",
        model=Groq(id=model),
        reasoning=True,
        instructions=[
            "You are a professional real-time intraday options trading assistant.",
            "You receive data from 3 independent price sources + technical indicators.",
            "First verify price consistency — flag UNRELIABLE if diff > ₹50.",
            "Consider: RSI, MACD, SMA, Bollinger Bands, market regime, macro news.",
            "If RSI < 30 → CALL bias. If RSI > 70 → PUT bias.",
            "If market is SIDEWAYS → suggest scalping with 5min/15min timeframe.",
            "Always give ONE clear signal: ENTER | HOLD | EXIT | WAIT",
            "For ENTER: exact ATM/ITM/OTM strike, approx premium, entry, SL, target, R:R.",
            "Give realistic win rate % and loss rate % with reasoning.",
            "Display everything in clean tables.",
            "Be concise — traders need fast decisions.",
        ],
        markdown=True,
    )


@st.cache_resource
def get_general_agent(model, technicals, news, fundamentals, history):
    return Agent(
        model=Groq(id=model),
        tools=[YFinanceTools(
            enable_stock_price=True,
            enable_analyst_recommendations=True,
            enable_company_info=True,
            enable_technical_indicators=technicals,
            enable_company_news=news,
            enable_stock_fundamentals=fundamentals,
            enable_historical_prices=history,
        )],
        instructions=[
            "You are an expert financial analyst.",
            "Always use tools to fetch real data.",
            "Display data in clean tables.",
            "Give clear BUY/SELL recommendations.",
            "Always use FULL symbol — never truncate ^NSEBANK.",
        ],
        markdown=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🤖 AI Trading Platform")
    mode = st.radio("Mode", ["💬 General Chat", "📡 Real-Time Trading"], index=1)
    model_choice = st.selectbox(
        "AI Brain",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
    )
    symbol = st.text_input("Symbol", value="^NSEBANK")
    st.divider()

    if mode == "💬 General Chat":
        st.markdown("### 🛠️ Tools")
        show_technicals   = st.toggle("Technical Indicators", True)
        show_news         = st.toggle("Company News",         True)
        show_fundamentals = st.toggle("Fundamentals",         True)
        show_history      = st.toggle("Historical Prices",    True)
        st.divider()
        st.markdown("### 💡 Quick Queries")
        if st.button("🏦 BankNifty Intraday"):
            st.session_state.query = f"Analyze {symbol} — BUY CALL or BUY PUT with strike, premium, entry, SL, target, win rate."
        if st.button("📊 NVIDIA Analysis"):
            st.session_state.query = "Analyze NVIDIA — price, fundamentals, analyst recommendations, news."
        if st.button("🇮🇳 Top Indian Banks"):
            st.session_state.query = "Compare HDFCBANK.NS, SBIN.NS, ICICIBANK.NS, KOTAKBANK.NS in a table."
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.markdown("### 📊 Position Tracker")
        pos = st.session_state.position
        if not pos["active"]:
            st.info("No active position")
            with st.expander("➕ Open Position"):
                pt  = st.selectbox("Type", ["CALL", "PUT"])
                sk  = st.number_input("Strike",    value=52000, step=100)
                en  = st.number_input("Entry",     value=52000, step=10)
                sl  = st.number_input("Stop Loss", value=52500, step=10)
                tg  = st.number_input("Target",    value=51000, step=10)
                pr  = st.number_input("Premium ₹", value=200,  step=10)
                qty = st.number_input("Lots",       value=1,    step=1, min_value=1)
                if st.button("✅ Open Position", type="primary"):
                    st.session_state.position = {
                        "active": True, "type": pt, "strike": sk,
                        "entry": en, "sl": sl, "target": tg,
                        "premium": pr, "qty": qty,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                    st.session_state.trade_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "action": "OPEN", "type": pt, "strike": sk,
                        "entry": en, "sl": sl, "target": tg, "premium": pr,
                    })
                    st.success("✅ Position opened!")
                    st.rerun()
        else:
            pnl_data = compute_pnl(pos, st.session_state.get("consensus_price") or pos["entry"])
            pnl_val  = pnl_data["option_pnl"] if pnl_data else 0
            pnl_cls  = "pnl-positive" if pnl_val >= 0 else "pnl-negative"
            st.markdown(f"""
            <div class='trade-card'>
            <b>🟢 {pos['type']} ACTIVE</b><br>
            Strike: ₹{pos['strike']} | Entry: ₹{pos['entry']}<br>
            SL: ₹{pos['sl']} | Target: ₹{pos['target']}<br>
            Premium: ₹{pos['premium']} | Lots: {pos.get('qty',1)}<br>
            Since: {pos['time']}<br>
            <span class='{pnl_cls}'>P&L: ₹{pnl_val:+.0f}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🔴 Close Position"):
                if pnl_data:
                    st.session_state.trade_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "action": "CLOSE", "type": pos['type'],
                        "strike": pos['strike'], "pnl": pnl_data["option_pnl"],
                    })
                st.session_state.position = {
                    "active": False, "type": None, "strike": None,
                    "entry": None, "sl": None, "target": None,
                    "premium": None, "time": None, "qty": 1
                }
                st.warning("Position closed!")
                st.rerun()

        st.divider()
        st.markdown("### 📜 Trade Log")
        if st.session_state.trade_log:
            for t in reversed(st.session_state.trade_log[-5:]):
                pnl_str = f" | P&L: ₹{t.get('pnl', 0):+.0f}" if 'pnl' in t else ""
                color = "#00e676" if t['action'] == "OPEN" else "#ff5252"
                st.markdown(
                    f"<span style='color:{color}'>{t['time']} {t['action']} "
                    f"{t['type']} {t['strike']}{pnl_str}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.caption("No trades yet")


# ============================================================
# MAIN — REAL-TIME TRADING
# ============================================================
if mode == "📡 Real-Time Trading":

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Dashboard", "🧠 AI Analysis",
        "📰 Macro News",     "📈 Charts & Indicators"
    ])

    # ── TAB 1: LIVE DASHBOARD ──────────────────────────────────
    with tab1:
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            fetch_btn = st.button("🔄 Fetch Live Data & Analyze", type="primary", use_container_width=True)
        with col_btn2:
            news_btn = st.button("📰 Fetch Macro News", use_container_width=True)
        with col_btn3:
            clear_btn = st.button("🗑️ Clear History", use_container_width=True)
            if clear_btn:
                st.session_state.messages = []
                st.rerun()

        if fetch_btn:
            with st.spinner("📡 Fetching from 3 sources..."):
                sources, consensus, diff, quality, yf_data = fetch_all(symbol)

            st.session_state.consensus_price = consensus

            # ── Price metrics ──────────────────────────────────
            st.subheader("💹 Live Price Feed")
            cols = st.columns(3)
            for i, s in enumerate(sources):
                with cols[i]:
                    if s.get("ok"):
                        delta_val = f"{s.get('change_pct', 0)}%"
                        st.metric(
                            f"{s['status']} {s['source']}",
                            f"₹{s.get('price', 'N/A')}",
                            delta_val,
                        )
                        h = s.get('high', 'N/A')
                        l = s.get('low',  'N/A')
                        st.caption(f"H: {h}  |  L: {l}")
                    else:
                        st.error(f"{s['source']}\n{s.get('error','')[:60]}")

            # Consensus
            if consensus:
                c1, c2, c3 = st.columns(3)
                c1.metric("📈 Consensus Price", f"₹{consensus}")
                c2.metric("📊 Data Quality",    quality)
                c3.metric("🕐 Last Update",     datetime.now().strftime("%H:%M:%S"))

            # ── Technical Indicators ───────────────────────────
            st.subheader("📉 Technical Indicators")

            hist   = yf_data.get("hist") if yf_data.get("ok") else None
            ind    = compute_indicators(hist)
            regime = compute_market_regime(ind, consensus)
            scalp  = get_scalping_timeframe(regime)

            if ind:
                ic1, ic2, ic3, ic4 = st.columns(4)

                # RSI
                with ic1:
                    rsi = ind.get("rsi")
                    rsi_color_hex = ind.get("rsi_color", "#7a9cc4")
                    st.markdown(f"""
                    <div class='indicator-card'>
                    <b>RSI (14)</b><br>
                    <span style='font-size:2rem;color:{rsi_color_hex}'>{rsi}</span><br>
                    <span style='font-size:0.8rem'>{ind.get('rsi_signal','N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # MACD
                with ic2:
                    macd_col = "#00e676" if ind.get("macd_signal") == "BUY" else "#ff5252"
                    st.markdown(f"""
                    <div class='indicator-card'>
                    <b>MACD</b><br>
                    <span style='font-size:1.4rem;color:{macd_col}'>{ind.get("macd_signal","N/A")}</span><br>
                    <span style='font-size:0.8rem'>MACD: {ind.get("macd","N/A")} | Sig: {ind.get("macd_signal_line","N/A")}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # SMA Trend
                with ic3:
                    trend_col = "#00e676" if "BULL" in (ind.get("trend") or "") else "#ff5252"
                    st.markdown(f"""
                    <div class='indicator-card'>
                    <b>Trend (SMA)</b><br>
                    <span style='font-size:1.4rem;color:{trend_col}'>{ind.get("trend","N/A")}</span><br>
                    <span style='font-size:0.8rem'>SMA20: {ind.get("sma20","N/A")} | SMA50: {ind.get("sma50","N/A")}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Market Regime
                with ic4:
                    regime_colors = {"SIDEWAYS": "#ffd600", "TRENDING": "#00e676", "VOLATILE": "#ff5252"}
                    rc = regime_colors.get(regime, "#7a9cc4")
                    st.markdown(f"""
                    <div class='indicator-card'>
                    <b>Market Regime</b><br>
                    <span style='font-size:1.4rem;color:{rc}'>{regime}</span><br>
                    <span style='font-size:0.8rem'>{scalp}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Bollinger Bands
                if ind.get("bb_upper") and consensus:
                    st.markdown(f"""
                    <div class='indicator-card'>
                    <b>Bollinger Bands</b> &nbsp;|&nbsp;
                    Lower: <span style='color:#00e676'>₹{ind['bb_lower']}</span> &nbsp;|&nbsp;
                    Price: <span style='color:#ffd600'>₹{consensus}</span> &nbsp;|&nbsp;
                    Upper: <span style='color:#ff5252'>₹{ind['bb_upper']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Smart Strike Selection ─────────────────────────
            if consensus and ind:
                st.subheader("🎯 Smart Strike Selection")
                rsi = ind.get("rsi")
                if rsi and rsi < 30:
                    suggested = "CALL"
                elif rsi and rsi > 70:
                    suggested = "PUT"
                else:
                    suggested = "CALL" if "BULL" in (ind.get("trend") or "") else "PUT"

                strikes = smart_strike(consensus, suggested)
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("📍 Suggested", suggested)
                sc2.metric("🎯 ATM Strike", f"₹{strikes['ATM']}")
                sc3.metric("💰 ITM Strike", f"₹{strikes['ITM']}")
                sc4.metric("🚀 OTM Strike", f"₹{strikes['OTM']}")

            # ── P&L Live ──────────────────────────────────────
            pos = st.session_state.position
            if pos["active"] and consensus:
                st.subheader("💰 Live P&L")
                pnl = compute_pnl(pos, consensus)
                if pnl:
                    p1, p2, p3, p4 = st.columns(4)
                    pnl_color = "#00e676" if pnl["option_pnl"] >= 0 else "#ff5252"
                    p1.metric("Index P&L",  f"₹{pnl['index_pnl']:+.2f}")
                    p2.metric("Option P&L", f"₹{pnl['option_pnl']:+.0f}")
                    p3.metric("P&L %",      f"{pnl['pnl_pct']:+.2f}%")
                    p4.metric("Lots × Size",f"{pnl['qty']} × {pnl['lot_size']}")

                    # SL/Target alerts
                    if pos["type"] == "PUT" and consensus >= pos["sl"]:
                        st.markdown("<div class='signal-exit'>🚨 STOP LOSS AT RISK — Consider EXIT</div>", unsafe_allow_html=True)
                    elif pos["type"] == "CALL" and consensus <= pos["sl"]:
                        st.markdown("<div class='signal-exit'>🚨 STOP LOSS AT RISK — Consider EXIT</div>", unsafe_allow_html=True)
                    elif pos["type"] == "PUT" and consensus <= pos["target"]:
                        st.markdown("<div class='signal-hold'>🎯 TARGET HIT — Book Profit!</div>", unsafe_allow_html=True)
                    elif pos["type"] == "CALL" and consensus >= pos["target"]:
                        st.markdown("<div class='signal-hold'>🎯 TARGET HIT — Book Profit!</div>", unsafe_allow_html=True)

            # ── AI ANALYSIS ────────────────────────────────────
            if consensus and ind:
                source_lines = "\n".join([
                    f"- {s['source']}: ₹{s.get('price')} | H={s.get('high')} L={s.get('low')} Chg={s.get('change_pct')}%"
                    for s in sources if s.get("ok")
                ])
                macro = st.session_state.get("macro_news", "Not fetched yet.")

                if pos["active"]:
                    prompt = f"""
LIVE DATA — {datetime.now().strftime('%d %b %Y %H:%M')} IST
3-Source Feed:
{source_lines}
Consensus: ₹{consensus} | Quality: {quality}

Technical Indicators:
RSI: {ind.get('rsi')} → {ind.get('rsi_signal')}
MACD Signal: {ind.get('macd_signal')} (MACD={ind.get('macd')}, Signal={ind.get('macd_signal_line')})
Trend: {ind.get('trend')} | SMA20={ind.get('sma20')} | SMA50={ind.get('sma50')}
Bollinger: Upper={ind.get('bb_upper')} Lower={ind.get('bb_lower')}
Market Regime: {regime} → {scalp}

Macro Context:
{macro}

ACTIVE POSITION:
{pos['type']} | Strike: ₹{pos['strike']} | Entry: ₹{pos['entry']}
SL: ₹{pos['sl']} | Target: ₹{pos['target']} | Since: {pos['time']}

Tasks:
1. Sources consistent? Flag if diff > ₹50
2. Market flipped against position?
3. SL at risk?
4. Signal: HOLD or EXIT with reason
5. Estimated P&L if exit now?
"""
                else:
                    prompt = f"""
LIVE DATA — {datetime.now().strftime('%d %b %Y %H:%M')} IST
3-Source Feed:
{source_lines}
Consensus: ₹{consensus} | Quality: {quality}

Technical Indicators:
RSI: {ind.get('rsi')} → {ind.get('rsi_signal')}
MACD Signal: {ind.get('macd_signal')} (MACD={ind.get('macd')}, Signal={ind.get('macd_signal_line')})
Trend: {ind.get('trend')} | SMA20={ind.get('sma20')} | SMA50={ind.get('sma50')}
Bollinger: Upper={ind.get('bb_upper')} Lower={ind.get('bb_lower')}
Market Regime: {regime} → {scalp}
Smart Strike (suggested {suggested}): ATM={strikes['ATM']} ITM={strikes['ITM']} OTM={strikes['OTM']}

Macro Context:
{macro}

NO ACTIVE POSITION.
Tasks:
1. Sources consistent? Flag if diff > ₹50
2. Consider RSI, MACD, SMA, BB, regime and macro news together
3. BUY CALL or BUY PUT?
4. Which strike — ATM/ITM/OTM and why?
5. Approx premium, entry, SL, target, R:R?
6. Win rate % and loss rate % with reasoning?
7. Best timeframe for this trade?
8. Signal: ENTER or WAIT?
"""
                with st.spinner("🧠 AI Reasoning..."):
                    try:
                        analysis = get_reasoning_agent(model_choice).run(prompt).content
                    except Exception as e:
                        analysis = f"AI error: {e}"

                st.session_state.last_analysis = analysis
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**[{datetime.now().strftime('%H:%M:%S')}]**\n\n{analysis}"
                })

                # Signal banner
                au = analysis.upper()
                if   "ENTER" in au and not pos["active"]:
                    st.markdown("<div class='signal-enter'>💡 ENTER SIGNAL — Open position in sidebar!</div>", unsafe_allow_html=True)
                elif "EXIT"  in au and pos["active"]:
                    st.markdown("<div class='signal-exit'>⚠️ EXIT SIGNAL — Close position in sidebar!</div>", unsafe_allow_html=True)
                elif "HOLD"  in au:
                    st.markdown("<div class='signal-hold'>🟢 HOLD — Stay in position</div>", unsafe_allow_html=True)
                elif "WAIT"  in au:
                    st.markdown("<div class='signal-wait'>⏳ WAIT — No setup yet</div>", unsafe_allow_html=True)

        else:
            st.info("👆 Click **Fetch Live Data & Analyze** to start")

    # ── TAB 2: AI ANALYSIS HISTORY ────────────────────────────
    with tab2:
        st.subheader("🧠 AI Analysis History")
        if st.session_state.messages:
            for msg in reversed(st.session_state.messages[-10:]):
                with st.expander(f"📋 {msg['content'][:80]}..."):
                    st.markdown(msg["content"])
        else:
            st.info("No analysis yet. Click 'Fetch Live Data & Analyze' on Dashboard tab.")

    # ── TAB 3: MACRO NEWS ─────────────────────────────────────
    with tab3:
        st.subheader("📰 Macro Market News")
        st.caption("RBI Policy | Fed News | Budget | Global Cues")

        if st.button("🔍 Fetch Latest Macro News", type="primary"):
            with st.spinner("📰 Searching latest news..."):
                try:
                    news_result = get_news_agent().run(
                        "Search for latest: RBI policy decision, US Fed interest rate news, "
                        "Indian budget impact on banking, global market cues affecting "
                        "Indian stock market today. Give 5 key points."
                    ).content
                    st.session_state["macro_news"] = news_result
                except Exception as e:
                    news_result = f"News fetch error: {e}"
                    st.session_state["macro_news"] = news_result

            st.markdown(news_result)

            # Category badges
            st.divider()
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.info("🏦 RBI Policy")
            bc2.info("🌍 Fed News")
            bc3.info("💰 Budget")
            bc4.info("🌐 Global Cues")
        else:
            if st.session_state.get("macro_news"):
                st.markdown(st.session_state["macro_news"])
            else:
                st.info("Click 'Fetch Latest Macro News' to get market-moving news")

    # ── TAB 4: CHARTS ─────────────────────────────────────────
    with tab4:
        st.subheader("📈 Price Charts & Technical Analysis")

        if st.button("📊 Load Charts", type="primary"):
            with st.spinner("Loading chart data..."):
                yf_data = get_yfinance_data(symbol)

            if yf_data.get("ok"):
                hist     = yf_data.get("hist")
                intraday = yf_data.get("intraday")
                ind      = compute_indicators(hist)

                ch1, ch2 = st.tabs(["📅 Daily Chart", "⚡ Intraday 5min"])

                with ch1:
                    if hist is not None and not hist.empty:
                        df_plot = ind.get("df", hist)
                        chart_data = df_plot[['Close', 'SMA20']].dropna()
                        st.line_chart(chart_data, height=350)

                        # RSI chart
                        if 'RSI' in df_plot.columns:
                            st.caption("RSI (14)")
                            rsi_data = df_plot[['RSI']].dropna()
                            st.line_chart(rsi_data, height=150)

                        # MACD chart
                        if 'MACD' in df_plot.columns:
                            st.caption("MACD")
                            macd_data = df_plot[['MACD', 'Signal']].dropna()
                            st.line_chart(macd_data, height=150)

                with ch2:
                    if intraday is not None and not intraday.empty:
                        st.caption("5-Minute Intraday Chart")
                        st.line_chart(intraday[['Close']], height=350)
                        vol_data = intraday[['Volume']].tail(50)
                        st.caption("Volume")
                        st.bar_chart(vol_data, height=150)
            else:
                st.error(f"Chart data unavailable: {yf_data.get('error')}")
        else:
            st.info("Click 'Load Charts' to view price charts")


# ============================================================
# MAIN — GENERAL CHAT
# ============================================================
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask anything about stocks, options, analysis...")

    if st.session_state.query:
        query = st.session_state.query
        st.session_state.query = ""

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                try:
                    agent  = get_general_agent(
                        model_choice,
                        show_technicals, show_news,
                        show_fundamentals, show_history,
                    )
                    result = agent.run(query).content
                    st.markdown(result)
                except Exception as e:
                    result = f"Error: {e}"
                    st.error(result)

        st.session_state.messages.append({"role": "assistant", "content": result})
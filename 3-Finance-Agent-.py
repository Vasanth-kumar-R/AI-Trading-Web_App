# realtime_trading_agent.py
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import requests
import yfinance as yf
import time
from datetime import datetime

load_dotenv()

# ============================================================
# DATA SOURCE 1: YFinance
# ============================================================
def get_yfinance_data():
    try:
        ticker = yf.Ticker("^NSEBANK")
        info = ticker.info
        hist = ticker.history(period="5d", interval="5m")
        last_5 = hist.tail(5)

        return {
            "source": "YFinance",
            "price": info.get("regularMarketPrice"),
            "open": info.get("regularMarketOpen"),
            "high": info.get("regularMarketDayHigh"),
            "low": info.get("regularMarketDayLow"),
            "volume": info.get("regularMarketVolume"),
            "prev_close": info.get("regularMarketPreviousClose"),
            "change_pct": round(info.get("regularMarketChangePercent", 0), 2),
            "week_high": info.get("fiftyTwoWeekHigh"),
            "week_low": info.get("fiftyTwoWeekLow"),
            "candles": last_5[['Open','High','Low','Close','Volume']].to_string()
        }
    except Exception as e:
        return {"source": "YFinance", "error": str(e)}


# ============================================================
# DATA SOURCE 2: NSE Direct (jugaad-data)
# ============================================================
def get_nse_data():
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        data = n.stock_quote_index("NIFTY BANK")
        price_info = data['priceInfo']
        return {
            "source": "NSE Direct",
            "price": price_info['lastPrice'],
            "open": price_info['open'],
            "high": price_info['intraDayHighLow']['max'],
            "low": price_info['intraDayHighLow']['min'],
            "prev_close": price_info['previousClose'],
            "change_pct": price_info['pChange'],
            "volume": data.get('marketDeptOrderBook', {}).get('totalBuyQuantity', 'N/A'),
        }
    except Exception as e:
        return {"source": "NSE Direct", "error": str(e)}


# ============================================================
# DATA SOURCE 3: Yahoo Finance API (Google Finance alternative)
# ============================================================
def get_google_data():
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEBANK"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        meta = data['chart']['result'][0]['meta']
        quotes = data['chart']['result'][0]['indicators']['quote'][0]

        highs = [h for h in quotes.get('high', []) if h]
        lows = [l for l in quotes.get('low', []) if l]

        return {
            "source": "Yahoo API",
            "price": meta.get("regularMarketPrice"),
            "open": meta.get("chartPreviousClose"),
            "high": max(highs) if highs else "N/A",
            "low": min(lows) if lows else "N/A",
            "prev_close": meta.get("previousClose"),
            "volume": meta.get("regularMarketVolume"),
            "change_pct": round(
                ((meta.get("regularMarketPrice", 0) - meta.get("previousClose", 1))
                 / meta.get("previousClose", 1)) * 100, 2
            ),
        }
    except Exception as e:
        return {"source": "Yahoo API", "error": str(e)}


# ============================================================
# DATA COMPARATOR
# ============================================================
def fetch_all_sources():
    print("\n📡 Fetching from all 3 sources...")
    sources = [
        get_yfinance_data(),
        get_nse_data(),
        get_google_data(),
    ]

    print("\n┌────────────────────┬───────────┬───────────┬───────────┬────────────┐")
    print("│ Source             │ Price     │ High      │ Low       │ Change%    │")
    print("├────────────────────┼───────────┼───────────┼───────────┼────────────┤")

    valid_prices = []
    for s in sources:
        if "error" not in s:
            price = s.get("price", "N/A")
            high  = s.get("high",  "N/A")
            low   = s.get("low",   "N/A")
            chg   = s.get("change_pct", "N/A")
            print(f"│ {s['source']:<18} │ {str(price):<9} │ {str(high):<9} │ {str(low):<9} │ {str(chg):<10} │")
            if price:
                valid_prices.append(float(price))
        else:
            err = s['error'][:50]
            print(f"│ {s['source']:<18} │ ❌ {err:<48} │")

    print("└────────────────────┴───────────┴───────────┴───────────┴────────────┘")

    if valid_prices:
        avg   = sum(valid_prices) / len(valid_prices)
        diff  = max(valid_prices) - min(valid_prices)
        status = "✅ CONSISTENT" if diff < 50 else "⚠️  DISCREPANCY!"
        print(f"\n📊 Consensus Price : ₹{avg:.2f}")
        print(f"📊 Max Difference  : ₹{diff:.2f}  →  {status}")
        return sources, avg, diff
    else:
        print("❌ All sources failed!")
        return sources, None, None


# ============================================================
# POSITION TRACKER
# ============================================================
class Position:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active       = False
        self.type         = None
        self.strike       = None
        self.entry_price  = None
        self.stop_loss    = None
        self.target       = None
        self.premium_paid = None
        self.entry_time   = None

    def open(self, pos_type, strike, entry, sl, target, premium):
        self.active       = True
        self.type         = pos_type.upper()
        self.strike       = float(strike)
        self.entry_price  = float(entry)
        self.stop_loss    = float(sl)
        self.target       = float(target)
        self.premium_paid = float(premium)
        self.entry_time   = datetime.now().strftime("%H:%M:%S")
        print(f"""
✅ Position Opened!
   Type     : {self.type}
   Strike   : ₹{self.strike}
   Entry    : ₹{self.entry_price}
   Stop Loss: ₹{self.stop_loss}
   Target   : ₹{self.target}
   Premium  : ₹{self.premium_paid}
   Time     : {self.entry_time}
        """)

    def close(self, reason, current_price):
        pnl = (current_price - self.entry_price
               if self.type == "CALL"
               else self.entry_price - current_price)
        net = pnl - self.premium_paid
        emoji = "🟢" if net > 0 else "🔴"
        print(f"""
{emoji} Position Closed!
   Reason   : {reason}
   Exit     : ₹{current_price:.2f}
   Index P&L: ₹{pnl:.2f}
   Net P&L  : ₹{net:.2f} (after premium)
        """)
        self.reset()

    def check_auto_exit(self, price):
        if not self.active:
            return None
        if self.type == "PUT"  and price <= self.target:
            return "TARGET HIT 🎯"
        if self.type == "CALL" and price >= self.target:
            return "TARGET HIT 🎯"
        if self.type == "PUT"  and price >= self.stop_loss:
            return "STOP LOSS HIT 🚨"
        if self.type == "CALL" and price <= self.stop_loss:
            return "STOP LOSS HIT 🚨"
        return None

    def status(self):
        if not self.active:
            return "No active position"
        return (f"{self.type} | Strike: ₹{self.strike} | "
                f"Entry: ₹{self.entry_price} | SL: ₹{self.stop_loss} | "
                f"Target: ₹{self.target} | Since: {self.entry_time}")


# ============================================================
# AGENTS
# ============================================================
data_agent = Agent(
    name="Data Fetcher",
    model=Groq(id="openai/gpt-oss-20b"),
    tools=[YFinanceTools(
        enable_stock_price=True,
        enable_technical_indicators=True,
        enable_historical_prices=True,
    )],
    instructions=[
        "Fetch technical indicators and 5-day historical prices for ^NSEBANK.",
        "Always use the FULL symbol ^NSEBANK — never truncate it.",
        "Return raw data only, no analysis.",
    ],
    markdown=False,
)

reasoning_agent = Agent(
    name="Trading Analyst",
    model=Groq(id="openai/gpt-oss-120b"),
    reasoning=True,
    instructions=[
        "You are a professional real-time intraday options trading assistant.",
        "You receive data from 3 independent sources — YFinance, NSE Direct, Yahoo API.",
        "First verify price consistency across all 3 sources.",
        "If discrepancy > ₹50 between any two sources — flag UNRELIABLE DATA and say WAIT.",
        "Always give ONE clear signal: ENTER | HOLD | EXIT | WAIT - win % should be =<80%",
        "For ENTER: give exact strike, premium, entry, stop loss, target, risk-reward, exact reason and safe positions",
        "For EXIT: give exact reason and estimated P&L.",
        "Give realistic win rate % and loss rate % with reasoning.",
        "Display everything in clean tables.",
        "Be concise — traders need fast clear decisions.",
    ],
    markdown=True,
)


# ============================================================
# MAIN REAL-TIME LOOP
# ============================================================
def run_realtime(interval=60):
    position = Position()

    print("""
╔══════════════════════════════════════════════════════════╗
║       🤖  AI Real-Time Trading Assistant                 ║
║       Sources : YFinance + NSE Direct + Yahoo API        ║
║       Models  : gpt-oss-20b (data) + gpt-oss-120b (brain)║
╚══════════════════════════════════════════════════════════╝
    """)

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now()

        # ── Market hours check ──────────────────────────────
        if now.hour < 9 or (now.hour == 9 and now.minute < 15):
            print("⏰ Market not open yet. Waiting...")
            time.sleep(60)
            continue

        if now.hour > 15 or (now.hour == 15 and now.minute >= 30):
            print("🔔 Market closed (after 3:30 PM IST). Stopping.")
            break

        print(f"\n{'='*62}")
        print(f"  🕐 [{now.strftime('%H:%M:%S')}]  Cycle #{cycle}")
        print(f"  📊 Position: {position.status()}")
        print(f"{'='*62}")

        # ── Step 1: Fetch all 3 sources ─────────────────────
        all_sources, consensus_price, price_diff = fetch_all_sources()

        if not consensus_price:
            print("❌ No valid data from any source. Retrying...")
            time.sleep(interval)
            continue

        # ── Step 2: Auto SL / Target check ──────────────────
        auto_exit = position.check_auto_exit(consensus_price)
        if auto_exit:
            print(f"\n🚨 {auto_exit} at ₹{consensus_price:.2f}")
            position.close(auto_exit, consensus_price)
            time.sleep(interval)
            continue

        # ── Step 3: Fetch technicals via YFinance agent ─────
        print("\n📉 Fetching technical indicators...")
        try:
            tech_resp   = data_agent.run(
                "Fetch 5-day technical indicators and historical OHLCV "
                "for ^NSEBANK. Include SMA, RSI if available."
            )
            technicals  = tech_resp.content
        except Exception as e:
            technicals  = f"Technical data unavailable: {e}"

        # ── Step 4: Build source summary ────────────────────
        source_lines = []
        for s in all_sources:
            if "error" not in s:
                source_lines.append(
                    f"- {s['source']}: Price=₹{s.get('price')} | "
                    f"High=₹{s.get('high')} | Low=₹{s.get('low')} | "
                    f"Change={s.get('change_pct')}%"
                )
            else:
                source_lines.append(f"- {s['source']}: ❌ {s['error']}")

        source_summary = "\n".join(source_lines)
        data_quality   = (
            "✅ All sources consistent"
            if price_diff and price_diff < 50
            else f"⚠️ Discrepancy ₹{price_diff:.2f} — treat with caution"
        )

        # ── Step 5: Build AI prompt ──────────────────────────
        if position.active:
            prompt = f"""
LIVE MARKET DATA — {now.strftime('%d %b %Y %H:%M:%S')} IST

3-Source Price Feed:
{source_summary}

Consensus Price : ₹{consensus_price:.2f}
Data Quality    : {data_quality}

Technical Indicators (YFinance):
{technicals}

ACTIVE POSITION:
{position.status()}

Your tasks:
1. Are all 3 sources consistent? Flag if discrepancy > ₹50.
2. Has the market flipped against my position?
3. Is my stop loss at risk of being hit soon?
4. Give signal: HOLD or EXIT with exact reason.
5. If EXIT — what is the estimated P&L?
6. Any key level to watch in the next 15 minutes?
"""
        else:
            prompt = f"""
LIVE MARKET DATA — {now.strftime('%d %b %Y %H:%M:%S')} IST

3-Source Price Feed:
{source_summary}

Consensus Price : ₹{consensus_price:.2f}
Data Quality    : {data_quality}

Technical Indicators (YFinance):
{technicals}

NO ACTIVE POSITION.

Your tasks:
1. Are all 3 sources consistent? Flag if discrepancy > ₹50.
2. Is there a high-probability intraday entry right now?
3. BUY CALL or BUY PUT — which one and why?
4. Exact strike price and approximate premium?
5. Entry price, stop loss level, target level?
6. Risk-reward ratio?
7. Win rate % and loss rate % with reasoning?
8. Signal: ENTER (with full details) or WAIT (with reason)?
"""

        # ── Step 6: AI Reasoning ─────────────────────────────
        print("\n🧠 AI Reasoning across all 3 sources...")
        try:
            response = reasoning_agent.run(prompt)
            analysis = response.content
            print(f"\n{analysis}")

            analysis_upper = analysis.upper()

            # EXIT detection
            if position.active and "EXIT" in analysis_upper:
                print("\n" + "⚠️ " * 10)
                print("  EXIT SIGNAL DETECTED!")
                print("⚠️ " * 10)
                user_input = input("\nPress E to EXIT position, any other key to HOLD: ").strip().upper()
                if user_input == "E":
                    position.close("AI Exit Signal", consensus_price)

            # ENTER detection
            elif not position.active and "ENTER" in analysis_upper:
                print("\n" + "💡 " * 10)
                print("  ENTER SIGNAL DETECTED!")
                print("💡 " * 10)
                user_input = input("\nEnter position? (Y/N): ").strip().upper()
                if user_input == "Y":
                    pos_type = input("CALL or PUT? ").strip().upper()
                    strike   = input("Strike price (e.g. 52000): ").strip()
                    entry    = input("Entry index price: ").strip()
                    sl       = input("Stop loss level: ").strip()
                    target   = input("Target level: ").strip()
                    premium  = input("Premium paid (₹): ").strip()
                    position.open(pos_type, strike, entry, sl, target, premium)

            elif "WAIT" in analysis_upper:
                print("\n⏳ Signal: WAIT — No trade setup right now.")

            elif position.active and "HOLD" in analysis_upper:
                print("\n🟢 Signal: HOLD — Stay in position.")

        except Exception as e:
            print(f"❌ AI Reasoning error: {e}")

        print(f"\n⏳ Next update in {interval}s... (Ctrl+C to stop)\n")
        time.sleep(interval)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("""
Choose update interval:
  60  = every 1 minute  (good for active trading)
  180 = every 3 minutes (balanced)
  300 = every 5 minutes (matches YFinance refresh)
    """)
    try:
        interval = int(input("Enter interval in seconds [default 60]: ").strip() or "60")
    except ValueError:
        interval = 60

    run_realtime(interval)
# 🖥️ Install Guide — Laptop/Desktop par kaise chalayein (Hinglish)

Ye software **100% local** chalta hai — tumhare laptop par. Koi data bahar
nahi jaata (sirf market data download hota hai aur AI ke liye OpenRouter call,
agar tum key daalte ho).

---

## Windows (sabse aasan) — 3 steps

1. **Python install karo** (ek baar):
   - https://www.python.org/downloads/ → "Download Python 3.12"
   - Installer me **"Add python.exe to PATH"** ka tick ✔ ZAROOR lagao
   - Install → Done

2. **Ye folder download karo** (Code → Download ZIP → extract) ya git clone:
   ```
   git clone https://github.com/roushan62/Trading-Software.git
   ```

3. **`installer/install_windows.bat` double-click karo.** Bas.
   - Ye khud virtual environment + saare packages install karega
   - **Desktop par "Trading Software" shortcut** bana dega
   - Double-click shortcut → browser me poora dashboard khul jayega 🎉

## macOS

```bash
bash installer/install_mac.sh
```
Desktop par **"Trading Software.command"** milega — double-click karo.
(Pehli baar pe Right-click → Open karna pad sakta hai security ke liye.)

## Linux

```bash
bash installer/install_linux.sh
./launch.sh
```

---

## Real market data laao (important!)

Default me demo (synthetic) data hota hai. Real data ke liye ek baar terminal
me (Windows me "cmd" kholo, folder me jao):

```
cd Trading-Software
.venv\Scripts\activate
python main.py fetch --symbols RELIANCE.NS,AAPL,MSFT,NVDA,BTC-USD,ETH-USD --timeframes 15m,1h,1d --provider yfinance
```

- **Indian stocks**: `RELIANCE.NS`, `TCS.NS`, `INFY.NS` (`.NS` = NSE, Yahoo format)
- **US stocks**: `AAPL`, `MSFT`, `NVDA`
- **Crypto**: `BTC-USD`, `ETH-USD` (24×7)
- Dashboard me `data/historical/` ke saare symbols automatically dikhte hain.

## AI assistant on karo (2 minute)

1. https://openrouter.ai par account banao (Google login chalta hai)
2. https://openrouter.ai/keys → **Create Key** → copy karo
3. App ke **sidebar me API key box me paste → "Save key"** dabao
   (key sirf tumhare laptop par `data/runtime/ai_key.txt` me save hoti hai —
   kabhi git/upload nahi hoti)
4. Model bhi change kar sakte ho sidebar me. Cheap + acche options:
   - `deepseek/deepseek-chat` (default, sasta + smart)
   - `openai/gpt-4o-mini` (fast)
   - `meta-llama/llama-3.1-70b-instruct` (free tier available)
   - `anthropic/claude-3.5-sonnet` (premium)

Bina key ke bhi app chalta hai — **offline rule-based advisor** concrete trade
plan deta hai (entry/SL/target ke saath). AI sirf better explanation deta hai.

---

## Roz ka usage

| Kya karna hai | Kahan |
|---|---|
| Market dekhna (candles, EMA, VWAP, S/R) | **📊 Market & Signal** tab |
| Multi-timeframe (15m/1h/1d ek saath) | Market tab ke neeche strip |
| Exact trade plan (kab, kahan entry/SL/target/qty) | Market tab → right panel |
| Aane wale 15 bars ka probability cone | **🔮 Forecast** tab |
| AI se poochna "kya abhi trade lena chahiye?" | **🤖 AI Assistant** tab |
| Open paper positions + journal | **📂 Positions & Journal** |
| Win rate / expectancy / drawdown | **📈 Stats** |

Live paper scanning (background me chalta re):
```
python main.py scan --symbols BTC-USD,RELIANCE.NS --timeframes 15m,1h --provider yfinance
```
Telegram alerts bhi hain — `alerts/telegram_bot.py` ke top par setup steps.

---

## ⚠️ Ekdum clear rakho

- **15-min "future chart" = probability cone hai, prediction nahi.** Koi bhi
  software (ya AI) future nahi bata sakta. Ye tool deta hai: statistics,
  probabilities, risk-defined plans.
- Har signal ka **stop-loss compulsory** hai — SL ke bina trade mat lo.
- Ye software **real orders place NAHI karta** — paper/decision-support only.
- **Not financial advice. Based on historical statistical edge, not a guarantee.**

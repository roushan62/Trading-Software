#!/bin/bash
# ============================================================
#  Market Analysis & Trade Signal Software - macOS Installer
#  Run:  bash installer/install_mac.sh
#  Creates a double-clickable "Trading Software.command" on Desktop
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo ""
echo "============================================"
echo "  Installing Trading Software (macOS)"
echo "============================================"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 nahi mila."
    echo "xcode-select --install ; phir 'brew install python' (https://brew.com)"
    exit 1
fi

echo "[1/4] Virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Packages (2-5 min)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/4] Demo data..."
python main.py fetch --symbols SYNTH --timeframes 1h --provider synthetic --bars 9000 >/dev/null 2>&1 || true

echo "[4/4] Desktop launcher..."
cat > "$HOME/Desktop/Trading Software.command" <<EOF
#!/bin/bash
cd "$(pwd)"
source .venv/bin/activate
python Launch.py
EOF
chmod +x "$HOME/Desktop/Trading Software.command"

echo ""
echo "============================================"
echo "  INSTALL HO GAYA!"
echo "============================================"
echo "  Desktop par 'Trading Software.command' double-click karo."
echo ""
echo "  REAL DATA:  python main.py fetch --symbols AAPL,BTC-USD --timeframes 15m,1h,1d --provider yfinance"
echo "  AI:        openrouter.ai/keys se key -> app sidebar me paste karo"
echo ""
echo "  Not financial advice. Based on historical statistical edge, not a guarantee."
echo ""

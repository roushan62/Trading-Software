#!/bin/bash
# ============================================================
#  Market Analysis & Trade Signal Software - Linux Installer
#  Run:  bash installer/install_linux.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo ""
echo "============================================"
echo "  Installing Trading Software (Linux)"
echo "============================================"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 nahi mila. sudo apt install python3-venv python3-pip"
    exit 1
fi

echo "[1/3] Virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/3] Packages (2-5 min)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "[3/3] Launcher script..."
cat > launch.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python Launch.py
EOF
chmod +x launch.sh
./launch.sh --seed-demo >/dev/null 2>&1 || python main.py fetch --symbols SYNTH --timeframes 1h --provider synthetic --bars 9000 >/dev/null 2>&1 || true

echo ""
echo "============================================"
echo "  INSTALL HO GAYA!"
echo "============================================"
echo "  Start karne ke liye:  ./launch.sh"
echo ""
echo "  REAL DATA:  python main.py fetch --symbols AAPL,BTC-USD --timeframes 15m,1h,1d --provider yfinance"
echo "  AI:        openrouter.ai/keys se key -> app sidebar me paste karo"
echo ""
echo "  Not financial advice. Based on historical statistical edge, not a guarantee."
echo ""

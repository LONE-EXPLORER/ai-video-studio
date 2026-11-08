#!/bin/bash
cd "$(dirname "$0")"

echo "🎬 Starting AI Video Studio..."

export PYTHONPATH="$PWD:$PYTHONPATH"

# Kill any lingering process on port 8501
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Activate virtual environment
source .venv/bin/activate

# Open browser after a brief delay
(sleep 2 && open http://localhost:8501) &

# Run Streamlit
exec streamlit run app/main.py --server.port 8501 --browser.gatherUsageStats false

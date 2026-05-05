#!/bin/bash
# One-command setup: install deps and launch the dashboard.
# Usage: bash run.sh
set -e
pip install -r requirements.txt
streamlit run src/dashboard/app.py

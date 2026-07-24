#!/bin/bash
# Run this script instead of clicking "Run All" manually.
# It runs the full notebook AND updates the HTML export together, every time.
#
# Usage (from the project root folder, in a terminal):
#   bash run_project.sh

set -e
cd "$(dirname "$0")/notebooks"

echo "Running the notebook..."
python3 -m nbconvert --to notebook --execute --inplace heart_disease_risk_prediction.ipynb

echo "Updating the HTML export..."
python3 -m nbconvert --to html heart_disease_risk_prediction.ipynb

echo ""
echo "Done! Both files are now up to date:"
echo "  - notebooks/heart_disease_risk_prediction.ipynb"
echo "  - notebooks/heart_disease_risk_prediction.html"

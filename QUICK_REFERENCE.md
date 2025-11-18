# 🚀 Quick Reference Card

## Setup (First Time Only)
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Run Applications

### Interactive Web App
```powershell
streamlit run llm_analysis_app.py
```
Opens: http://localhost:8501

### Generate Reports
```powershell
python generate_llm_reports.py
```
Creates: llm_reports/ folder with 5 reports

## Files
- `lstm_gry.ipynb` - LSTM/GRU models
- `projet R.R` - R analysis (ARIMA/GARCH)
- `llm_analysis_app.py` - Streamlit app
- `generate_llm_reports.py` - Report generator
- `NVDA.csv` - Stock data
- `.env` - API key (configured)

## Notes
- API key is in .env (no need to change)
- Virtual environment in venv/ folder
- Generated reports go to llm_reports/
- All 4 LLM points covered ✅

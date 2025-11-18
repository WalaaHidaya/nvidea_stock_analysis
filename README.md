# 📈 NVIDIA Stock Analysis with LLM

Complete stock analysis project combining LSTM/GRU models with LLM-powered insights for NVIDIA (NVDA) stock prediction.

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis

**Option A: Interactive Web App** (Recommended)
```powershell
streamlit run llm_analysis_app.py
```

**Option B: Generate All Reports**
```powershell
python generate_llm_reports.py
```

## 📁 Project Files

- `lstm_gry.ipynb` - LSTM/GRU model training & evaluation
- `projet R.R` - Classical models (ARIMA, SARIMA, GARCH)
- `llm_analysis_app.py` - Interactive Streamlit application
- `generate_llm_reports.py` - Automated report generator
- `NVDA.csv` - Stock data from Yahoo Finance

## LLM Features 

The project uses **Qwen 3-32B** via Groq API to:

1. **Generate model hypotheses** from descriptive statistics
2. **Explain results** in simplified language
3. **Provide investment recommendations** with explicit risks & limits
4. **Compare human vs AI** explanations for coherence

## 📊 Generated Reports

After running `generate_llm_reports.py`, find reports in `llm_reports/`:
- `0_RAPPORT_COMPLET_CONSOLIDE.txt` - Complete consolidated report
- `1_hypotheses_modeles.txt` - Model hypotheses
- `2_explication_resultats.txt` - Simplified explanations
- `3_recommandations_investissement.txt` - Investment recommendations
- `4_comparaison_humain_vs_ia.txt` - Human vs AI comparison

## ⚙️ Configuration

Groq API key is stored in `.env` file.

## ⚠️ Important Disclaimers

- Generated recommendations are for **educational purposes only**
- **Not financial advice** - consult a licensed advisor
- AI cannot predict unpredictable events (announcements, crises)
- Past performance doesn't guarantee future results

---

**Model**: Qwen 3-32B via Groq API  
**Data**: NVIDIA (NVDA) 2020-Present  
**Technologies**: Python, Streamlit, TensorFlow, R

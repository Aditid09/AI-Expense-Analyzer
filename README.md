# AI Expense Behavior Analyzer

This version adds **Manual Financial Data Entry**: enter monthly salary/income and category-wise spending, save it into the current session, review manual-entry history, and download current data as CSV.

Other features include AI categorization, KMeans behavior profiling, Isolation Forest anomaly detection, forecasting, smart budgeting, recurring/duplicate detection, dashboard charts, CSV/Excel upload, and an AI-style assistant.

## Windows
```powershell
cd AI_Expense_Behavior_Analyzer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

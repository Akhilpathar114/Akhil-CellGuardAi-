# CellGuard.AI - Streamlit App (Final Package)

This zip contains:
- `app.py` : Streamlit dashboard with scenario demo data, unique chart keys to prevent StreamlitDuplicateElementId, PDF export, and many diagnostic charts.
- `requirements.txt` : Python packages needed to run the app.
- `README.md` : This file.

## How to run locally

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate  # Windows PowerShell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## Notes
- If ReportLab wheel fails on your environment, try removing the version pin in requirements and reinstall.
- The app includes scenario presets (Generic, EV, Drone, Phone) selectable in the sidebar.

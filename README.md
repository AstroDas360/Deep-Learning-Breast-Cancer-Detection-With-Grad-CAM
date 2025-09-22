# Breast Cancer Detection (Mini-DDSM)

This repo contains a single master notebook that trains a CNN (VGG16 by default) to detect malignancy on Mini-DDSM, produces explainability overlays (Grad-CAM), and exports a Streamlit UI.


## Structure
- `MINI-DDSM-Complete-PNG-16/` - can be downloaded from kaggle 
- `Data.xlsx` – metadata with paths, status, view, mask info
- `notebook/b_cancer_final_v4.ipynb` – main workflow
- `models/` – contains all models that were trained
- `outputs/` – metrics CSVs, plots, Grad-CAM panels, visual quality check
- `streamlit_app/` – UI files 
- `requirements.txt` - pip install

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run app
```bash
source .venv/bin/activate                   
python -m streamlit run streamlit_app/app.py
```
# Deep-Learning-Breast-Cancer-Detection with Grad-CAM
CM3070 Final Project Codebase

Refer to the main codes in the following paths:
/workspaces/Deep-Learning-Breast-Cancer-Detection/notebooks/b_cancer_final_v6.ipynb
/workspaces/Deep-Learning-Breast-Cancer-Detection/streamlit_app/app.py
/workspaces/Deep-Learning-Breast-Cancer-Detection/streamlit_app/app_bundle_v6/preprocess.py
/workspaces/Deep-Learning-Breast-Cancer-Detection/streamlit_app/app_bundle_v6/gradcam_utils.py


## Structure
- `MINI-DDSM-Complete-PNG-16/` - downloaded from kaggle
- `Data.xlsx` – metadata with paths, status, view, mask info
- `notebook/b_cancer_final_v4.ipynb` – main workflow
- `models/` – contains all models that were trained
- `outputs/` – metrics CSVs, plots, Grad-CAM panels, visual quality check
- `streamlit_app/` – UI files 
- `requirements.txt` - pip install

### Note: due to the upload size limit, I am unable to upload the rest of my files: the dataset, the models and saved images.

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

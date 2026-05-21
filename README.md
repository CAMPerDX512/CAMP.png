# CAMP.png — Streamlit Web App

Deploy in 5 minutes with no server, no Docker, no command line.

---

## Deploy to Streamlit Community Cloud (free)

### Step 1 — Push to GitHub
Upload all files in this folder to a GitHub repository
(either your existing CAMP.png repo or a new one).

Required files:
```
app.py
requirements.txt
.streamlit/config.toml
analysis/
  __init__.py
  _registry.py
  pet_uniformity.py
  pet_resolution.py
  acr_ct_phantom.py
core/
  __init__.py
  dicom_loader.py
```

### Step 2 — Create a Streamlit account
Go to https://share.streamlit.io and sign in with GitHub.

### Step 3 — Deploy
1. Click **"New app"**
2. Select your GitHub repo
3. Set the **Main file path** to `app.py`
4. Click **Deploy**

Streamlit installs all packages from `requirements.txt` automatically.
Your app will be live at:
```
https://YOUR_NAME-REPO_NAME-app-HASH.streamlit.app
```

### Step 4 — Custom URL (optional)
In your app settings on Streamlit Cloud you can set a custom subdomain like:
```
https://camppng.streamlit.app
```

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App opens at http://localhost:8501

---

## Adding new analysis scripts

Drop a `.py` file into `analysis/` with:
```python
DISPLAY_NAME = "My Analysis"
DESCRIPTION  = "What it does"

def run(images_sorted, metadata, progress_cb=None):
    ...
    return {"dataframe": df, "figures": [fig1]}
```

Commit and push — the app reloads automatically on Streamlit Cloud.

---

## File size limits

Streamlit Community Cloud allows uploads up to 500 MB per file.
For larger datasets, increase `maxUploadSize` in `.streamlit/config.toml`.

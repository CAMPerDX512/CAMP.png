# CAMP.png — Vendored GitHub Pages Deployment

This folder contains everything needed to run CAMP.png as a fully
self-hosted GitHub Pages site. All Python packages load from your
repo — no external CDN calls at runtime.

---

## One-time setup (do this once on your local machine)

### 1. Install Python if you don't have it
https://python.org

### 2. Run the download script
```bash
python download_pyodide_assets.py
```

This downloads ~70 MB of Pyodide wheels + runtime into a `pyodide/`
folder. Takes 1–3 minutes depending on your connection.

### 3. Confirm the pyodide/ folder was created
You should see files like:
```
pyodide/
  pyodide.js
  pyodide.asm.wasm
  numpy-1.26.4-...whl
  scipy-1.13.0-...whl
  pandas-2.2.2-...whl
  scikit_image-0.23.2-...whl
  pydicom-2.4.4-py3-none-any.whl
  fflate.min.js
  ... (20+ files total)
```

---

## Deploying to GitHub Pages

### Option A — Update an existing repo
If you already have a CAMP.png GitHub Pages repo:

```bash
# Copy these two items into your repo folder
cp index.html  /path/to/your/repo/index.html
cp -r pyodide/ /path/to/your/repo/pyodide/

cd /path/to/your/repo
git add index.html pyodide/
git commit -m "Vendor Pyodide packages for fast local loading"
git push
```

### Option B — Fresh repo
```bash
git init camppng
cd camppng
cp /path/to/index.html .
cp -r /path/to/pyodide .
git add .
git commit -m "Initial CAMP.png deploy"
git remote add origin https://github.com/YOUR_NAME/camppng.git
git push -u origin main
```
Then enable GitHub Pages: Settings → Pages → Deploy from branch → main / root.

---

## Why this is faster

Before (CDN loading):
- Browser requests each wheel from cdn.jsdelivr.net at runtime
- Cold start: 30–60 seconds (sequential downloads + install)
- Warm start: still 10–20 seconds (CDN cache varies by location)

After (vendored):
- GitHub serves wheels from the same origin as the page
- GitHub Pages uses a global CDN (Fastly) — files are edge-cached
- Cold start: ~8–15 seconds (parallel loads from one fast origin)
- Warm start: ~2–5 seconds (browser cache serves wheels instantly)
- Repeat visitors: near-instant (wheels are large but stable — cached for weeks)

---

## Folder structure after setup
```
your-repo/
├── index.html               ← the app (single file)
├── download_pyodide_assets.py  ← run once to populate pyodide/
├── pyodide/
│   ├── pyodide.js           ← Pyodide runtime loader
│   ├── pyodide.asm.wasm     ← WebAssembly binary (~7 MB)
│   ├── python_stdlib.zip    ← Python standard library
│   ├── pyodide-lock.json    ← package index
│   ├── numpy-*.whl
│   ├── scipy-*.whl
│   ├── pandas-*.whl
│   ├── scikit_image-*.whl
│   ├── Pillow-*.whl
│   ├── pydicom-*.whl
│   ├── fflate.min.js
│   └── ... (dependency wheels)
└── README.md
```

---

## Updating the app

To update the HTML only (no package changes):
```bash
cp new_index.html index.html
git add index.html && git commit -m "Update app" && git push
```

To update packages (rare):
1. Edit `download_pyodide_assets.py` with new version numbers
2. Delete the old `pyodide/` folder
3. Re-run `python download_pyodide_assets.py`
4. Commit and push the updated `pyodide/` folder

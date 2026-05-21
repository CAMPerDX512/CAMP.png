"""
CAMP.png — Streamlit Web Application
DICOM Viewer & Analysis
"""

import io
import os
import sys
import tempfile
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from core.dicom_loader import load_bytes_as_series
from analysis._registry import discover

ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis")


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CAMP.png",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark theme overrides */
[data-testid="stAppViewContainer"] { background: #0f0f0f; }
[data-testid="stSidebar"] { background: #141414; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
h1,h2,h3 { color: #e2e2ea; }
.stButton > button {
    background: #1e3a5f; color: #7ab8ff;
    border: 1px solid #2a5080; border-radius: 6px;
    width: 100%;
}
.stButton > button:hover { background: #264f8a; }

/* Header bar */
.camp-header {
    background: linear-gradient(180deg,#060d1f 0%,#0e2248 55%,#1a3d7a 100%);
    padding: 16px 24px 10px;
    border-bottom: 1px solid #1a3a6a;
    margin: -1rem -1rem 1rem -1rem;
    display: flex; align-items: center; gap: 16px;
}
.camp-title { font-size: 26px; font-weight: 700; color: #fff;
    letter-spacing: 1px; margin: 0; }
.camp-sub { font-size: 10px; color: rgba(160,200,240,.7);
    letter-spacing: 3px; margin: 2px 0 0; }
.anno-box {
    background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
    padding: 8px 10px; font-family: 'Courier New', monospace;
    font-size: 11px; line-height: 1.6; color: #ccc; margin-bottom: 8px;
    white-space: pre-wrap;
}
.result-card {
    background: #1e1e1e; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 12px 14px; margin-bottom: 8px;
}
.metric-big { font-size: 22px; font-weight: 700; color: #4a9eff; }
.metric-label { font-size: 10px; color: #666; letter-spacing: 1px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# State helpers
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "series_map":     {},   # label → {images, infos, metadata}
        "active_series":  None,
        "slice_idx":      0,
        "ww":             None,
        "wl":             None,
        "show_anno":      True,
        "analyses":       None,  # cached list of (name, desc, module)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# DICOM helpers
# ══════════════════════════════════════════════════════════════════════════════
def _windowed_png(arr: np.ndarray, ww: float, wl: float) -> Image.Image:
    lo = wl - ww / 2; hi = wl + ww / 2
    clipped = np.clip(arr, lo, hi)
    norm = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(norm, "L").convert("RGB")


def _default_wl(arr: np.ndarray):
    mn, mx = float(np.min(arr)), float(np.max(arr))
    return max(mx - mn, 1.0), (mn + mx) / 2


def _get_tag(ds, attr, default=""):
    try:
        v = getattr(ds, attr, None)
        if v is None: return default
        if hasattr(v, "__len__") and not isinstance(v, str):
            v = v[0] if len(v) > 0 else default
        return str(v).strip() or default
    except Exception:
        return default


def _get_tagf(ds, attr):
    try:
        v = getattr(ds, attr, None)
        if v is None: return None
        if hasattr(v, "__len__") and not isinstance(v, str): v = v[0]
        return float(v)
    except Exception:
        return None


def _fmt_date(d): return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d)==8 and d.isdigit() else d
def _fmt_time(t): t=t.split(".")[0]; return f"{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t)>=6 else t


def _build_annotations(ds, slice_idx: int, n_slices: int) -> dict:
    if ds is None:
        return {"TL": [], "TR": [], "BL": [], "BR": []}
    g = _get_tag; gf = _get_tagf

    name = str(g(ds,"PatientName","Unknown")).replace("^"," ").strip()
    tl = [name]
    for a, l in [("PatientID","ID"),("PatientBirthDate","DOB"),("PatientSex","Sex")]:
        v = g(ds, a)
        if v: tl.append(f"{l}: {_fmt_date(v) if 'Date' in a else v}")
    sd = g(ds,"StudyDate"); st2 = g(ds,"StudyTime")
    if sd: tl.append(f"Date: {_fmt_date(sd)}  {_fmt_time(st2)}".rstrip())
    for a in ("StudyDescription","AccessionNumber"):
        v = g(ds, a)
        if v: tl.append(v if a=="StudyDescription" else f"Acc: {v}")

    mod = g(ds,"Modality","")
    tr = []
    for a in ("InstitutionName",):
        v = g(ds, a)
        if v: tr.append(v)
    mfr = g(ds,"Manufacturer"); mdl = g(ds,"ManufacturerModelName")
    if mfr or mdl: tr.append(f"{mfr} {mdl}".strip())
    for a, l in [("Modality","Modality"),("SeriesDescription",""),
                 ("SeriesNumber","Series"),("ProtocolName","Protocol")]:
        v = g(ds, a)
        if v: tr.append(v if not l else f"{l}: {v}")
    if mod in ("CT",""):
        for a, l in [("KVP","kVp"),("XRayTubeCurrent","mA"),("Exposure","mAs"),("CTDIvol","CTDIvol")]:
            v = gf(ds, a)
            if v is not None: tr.append(f"{l}: {v:.0f}" + (" mGy" if a=="CTDIvol" else ""))
        for a, l in [("SpiralPitchFactor","Pitch"),("RevolutionTime","Rot time")]:
            v = gf(ds, a)
            if v is not None and 0.1 < v < 10: tr.append(f"{l}: {v:.2f} s" if "time" in l else f"{l}: {v:.2f}")
    elif mod == "MR":
        for a, l, u in [("MagneticFieldStrength","Field"," T"),("RepetitionTime","TR"," ms"),
                        ("EchoTime","TE"," ms"),("FlipAngle","Flip","°")]:
            v = gf(ds, a)
            if v is not None: tr.append(f"{l}: {v:.1f}{u}")
    elif mod in ("PT","NM"):
        try:
            rp = ds.RadiopharmaceuticalInformationSequence[0]
            radio = str(getattr(rp,"Radiopharmaceutical","")).strip()
            dose  = float(getattr(rp,"RadionuclideTotalDose",0))
            if radio: tr.append(f"Tracer: {radio}")
            if dose: tr.append(f"Dose: {dose/1e6:.1f} MBq")
        except Exception: pass

    ps = getattr(ds,"PixelSpacing",None)
    thick = gf(ds,"SliceThickness"); loc = gf(ds,"SliceLocation")
    bl = [f"Slice: {slice_idx+1} / {n_slices}"]
    if loc   is not None: bl.append(f"Loc: {loc:.2f} mm")
    if thick is not None: bl.append(f"Thickness: {thick:.2f} mm")
    rows = g(ds,"Rows"); cols = g(ds,"Columns")
    if rows and cols: bl.append(f"Matrix: {rows} × {cols}")
    if ps:
        try: bl.append(f"Pixel: {float(ps[0]):.3f} × {float(ps[1]):.3f} mm")
        except Exception: pass
    if ps and rows and cols:
        try: bl.append(f"FOV: {float(ps[0])*int(rows):.0f} × {float(ps[1])*int(cols):.0f} mm")
        except Exception: pass

    br = []
    it = g(ds,"ImageType")
    if it: br.append(it.replace("\\","/ ")[:30])
    for a, l in [("ConvolutionKernel","Kernel"),("PatientPosition","Position")]:
        v = g(ds, a)
        if v: br.append(f"{l}: {v}")
    rd = gf(ds,"ReconstructionDiameter")
    if rd: br.append(f"Recon Ø: {rd:.0f} mm")

    return {"TL": tl, "TR": tr, "BL": bl, "BR": br}


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="camp-header">
  <div>
    <p class="camp-title">⛰️ CAMP.png</p>
    <p class="camp-sub">DICOM VIEWER &amp; ANALYSIS</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Series management & file upload
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Load DICOM Files")
    uploaded = st.file_uploader(
        "Drop .dcm, .ima or .zip files",
        type=["dcm","ima","zip"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Parsing DICOM files…"):
            pairs = []
            for f in uploaded:
                data = f.read()
                if f.name.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(data)) as z:
                            for name in z.namelist():
                                if not name.endswith("/"):
                                    pairs.append((os.path.basename(name), z.read(name)))
                    except Exception:
                        pairs.append((f.name, data))
                else:
                    pairs.append((f.name, data))
            try:
                new_groups = load_bytes_as_series(pairs)
                added = 0
                for label, series in new_groups.items():
                    if label not in st.session_state.series_map:
                        st.session_state.series_map[label] = series
                        added += 1
                if added:
                    st.success(f"Loaded {added} new series")
                    if st.session_state.active_series is None:
                        first = list(new_groups.keys())[0]
                        st.session_state.active_series = first
                        n = new_groups[first]["metadata"]["n_slices"]
                        st.session_state.slice_idx = n // 2
                        arr = new_groups[first]["images"][n // 2]
                        ww, wl = _default_wl(arr)
                        st.session_state.ww = ww
                        st.session_state.wl = wl
                    st.rerun()
            except Exception as e:
                st.error(f"Load failed: {e}")

    # Series list
    st.markdown("---")
    st.markdown("### 🗂️ Series")
    series_map = st.session_state.series_map

    if not series_map:
        st.caption("No series loaded yet.")
    else:
        for label, series in list(series_map.items()):
            meta = series["metadata"]
            cols2 = st.columns([6, 1])
            with cols2[0]:
                active = (label == st.session_state.active_series)
                btn_label = f"{'▶ ' if active else ''}{meta['series_desc'] or label}\n{meta['modality']} · {meta['n_slices']} sl"
                if st.button(btn_label, key=f"sel_{label}", use_container_width=True):
                    st.session_state.active_series = label
                    n = meta["n_slices"]
                    st.session_state.slice_idx = n // 2
                    arr = series["images"][n // 2]
                    ww, wl = _default_wl(arr)
                    st.session_state.ww = ww
                    st.session_state.wl = wl
                    st.rerun()
            with cols2[1]:
                if st.button("✕", key=f"rm_{label}", help="Remove series"):
                    del st.session_state.series_map[label]
                    if st.session_state.active_series == label:
                        remaining = list(st.session_state.series_map.keys())
                        st.session_state.active_series = remaining[0] if remaining else None
                    st.rerun()

        if st.button("✕ Clear all series", use_container_width=True):
            st.session_state.series_map = {}
            st.session_state.active_series = None
            st.rerun()

    # Window / Level controls
    st.markdown("---")
    st.markdown("### 🔆 Window / Level")
    active_label = st.session_state.active_series
    if active_label and active_label in series_map:
        series = series_map[active_label]
        images = series["images"]
        arr = images[st.session_state.slice_idx]
        mn, mx = float(np.min(arr)), float(np.max(arr))
        rng = max(mx - mn, 1.0)
        ww_val = st.slider("Window width",  1.0, rng * 2, float(st.session_state.ww or rng), step=rng/200, format="%.0f")
        wl_val = st.slider("Window level", mn - rng * 0.2, mx + rng * 0.2, float(st.session_state.wl or (mn+mx)/2), step=rng/200, format="%.0f")
        if ww_val != st.session_state.ww or wl_val != st.session_state.wl:
            st.session_state.ww = ww_val
            st.session_state.wl = wl_val
        if st.button("Reset W/L"):
            ww, wl = _default_wl(arr)
            st.session_state.ww = ww
            st.session_state.wl = wl
            st.rerun()

    st.session_state.show_anno = st.toggle("Show annotations", value=st.session_state.show_anno)


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════
active_label = st.session_state.active_series

if not active_label or active_label not in series_map:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#444">
      <div style="font-size:64px;margin-bottom:16px">⛰️</div>
      <div style="font-size:18px;margin-bottom:8px;color:#666">No series loaded</div>
      <div style="font-size:13px">Upload .dcm, .ima, or .zip files using the sidebar</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

series   = series_map[active_label]
images   = series["images"]
infos    = series["infos"]
metadata = series["metadata"]
n_slices = len(images)

# ── Viewer + Analysis columns ─────────────────────────────────────────────────
viewer_col, analysis_col = st.columns([3, 1])

with viewer_col:
    # Slice slider
    idx = st.slider("Slice", 0, n_slices - 1, st.session_state.slice_idx,
                    format="%d", label_visibility="collapsed",
                    key="slice_slider")
    if idx != st.session_state.slice_idx:
        st.session_state.slice_idx = idx
    slice_idx = st.session_state.slice_idx

    arr = images[slice_idx]
    ds  = infos[slice_idx] if slice_idx < len(infos) else None
    ww  = st.session_state.ww or _default_wl(arr)[0]
    wl  = st.session_state.wl or _default_wl(arr)[1]

    img_pil = _windowed_png(arr, ww, wl)
    st.image(img_pil, use_container_width=True, caption=f"Slice {slice_idx+1} / {n_slices}  |  W:{ww:.0f}  L:{wl:.0f}")

    # Annotations
    if st.session_state.show_anno and ds is not None:
        anno = _build_annotations(ds, slice_idx, n_slices)
        a1, a2 = st.columns(2)
        with a1:
            if anno["TL"]:
                st.markdown(f'<div class="anno-box">' + "\n".join(anno["TL"]) + "</div>", unsafe_allow_html=True)
            if anno["BL"]:
                st.markdown(f'<div class="anno-box">' + "\n".join(anno["BL"]) + "</div>", unsafe_allow_html=True)
        with a2:
            if anno["TR"]:
                st.markdown(f'<div class="anno-box">' + "\n".join(anno["TR"]) + "</div>", unsafe_allow_html=True)
            br_lines = [f"W: {ww:.0f}  L: {wl:.0f}"] + anno["BR"]
            st.markdown(f'<div class="anno-box">' + "\n".join(br_lines) + "</div>", unsafe_allow_html=True)


# ── Analysis panel ────────────────────────────────────────────────────────────
with analysis_col:
    st.markdown("#### 🔬 Analysis")

    if st.session_state.analyses is None:
        st.session_state.analyses = discover(ANALYSIS_DIR)

    analyses = st.session_state.analyses
    if not analyses:
        st.caption("No analysis modules found.")

    for name, desc, module in analyses:
        with st.expander(f"**{name}**", expanded=False):
            st.caption(desc)
            needs_ds = getattr(module, "NEEDS_DATASETS", False) or "acr" in name.lower()
            data = infos if (needs_ds and infos) else images

            if st.button(f"▶ Run {name}", key=f"run_{name}", use_container_width=True):
                progress_bar = st.progress(0, text="Running…")
                status_text  = st.empty()
                try:
                    result = module.run(
                        data, metadata,
                        progress_cb=lambda v, pb=progress_bar: pb.progress(v, text=f"{v}%")
                    )
                    progress_bar.progress(100, text="Done")

                    # Store result in session state
                    st.session_state[f"result_{name}"] = result

                    # Show summary metrics
                    inter = result.get("interslice")
                    intra = result.get("intraslice")
                    if inter is not None and not (isinstance(inter, float) and np.isnan(inter)):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Interslice", f"{inter:.2f}%")
                        with c2:
                            st.metric("Intraslice", f"{intra:.2f}%")

                    n_src = result.get("n_sources")
                    if n_src is not None:
                        st.metric("Sources found", n_src)

                    errs = result.get("errors", {})
                    if isinstance(errs, dict) and errs:
                        for mod_name, tb in errs.items():
                            st.warning(f"{mod_name} failed — see details below")
                            with st.expander(f"{mod_name} traceback"):
                                st.code(tb, language="python")

                except Exception as e:
                    import traceback
                    progress_bar.empty()
                    st.error(f"Analysis failed: {e}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc(), language="python")

            # Show cached results if available
            cached = st.session_state.get(f"result_{name}")
            if cached:
                # DataFrames
                for key in ("dataframe","df_module1","df_module2","df_module3","df_module4"):
                    df = cached.get(key)
                    if df is not None and not df.empty:
                        lbl = {"dataframe":"Results","df_module1":"Module 1",
                               "df_module2":"Module 2","df_module3":"Module 3",
                               "df_module4":"Module 4"}.get(key, key)
                        st.caption(lbl)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                # Figures
                figs = cached.get("figures", [])
                if figs:
                    tab_labels = [f"Fig {i+1}" for i in range(len(figs))]
                    tabs = st.tabs(tab_labels)
                    for i, (tab, fig) in enumerate(zip(tabs, figs)):
                        with tab:
                            st.pyplot(fig, use_container_width=True)

                # Export
                dfs_to_export = {k: cached[k] for k in
                                 ("dataframe","df_module1","df_module2","df_module3","df_module4")
                                 if k in cached and cached[k] is not None and not cached[k].empty}
                if dfs_to_export:
                    buf = io.BytesIO()
                    with __import__("pandas").ExcelWriter(buf, engine="openpyxl") as writer:
                        for sheet_key, df in dfs_to_export.items():
                            sn = {"dataframe":"Results","df_module1":"Module1",
                                  "df_module2":"Module2","df_module3":"Module3",
                                  "df_module4":"Module4"}.get(sheet_key, sheet_key)
                            df.to_excel(writer, sheet_name=sn, index=False)
                    st.download_button(
                        "⬇ Export to Excel",
                        data=buf.getvalue(),
                        file_name=f"{name.replace(' ','_')}_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"export_{name}",
                    )

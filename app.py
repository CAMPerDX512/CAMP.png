"""
CAMP.png — Streamlit Web Application
DICOM Viewer & Analysis
"""

import io
import os
import sys
import zipfile
import traceback

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dicom_loader import load_bytes_as_series
from analysis._registry import discover

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")

# ══════════════════════════════════════════════════════════════════════════════
# Page config — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CAMP.png",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f0f0f; }
[data-testid="stSidebar"]          { background: #141414; }
h1,h2,h3,h4                        { color: #e2e2ea !important; }
.stButton > button {
    background: #1e3a5f; color: #7ab8ff;
    border: 1px solid #2a5080; border-radius: 6px; width: 100%;
}
.stButton > button:hover { background: #264f8a; border-color:#4a9eff; }
.anno-box {
    background:#1a1a1a; border:1px solid #2a2a2a; border-radius:5px;
    padding:8px 10px; font-family:'Courier New',monospace;
    font-size:11px; line-height:1.6; color:#ccc; white-space:pre-wrap;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    for k, v in {
        "series_map":    {},
        "active_series": None,
        "slice_idx":     0,
        "ww":            None,
        "wl":            None,
        "show_anno":     True,
        "analyses":      [],
        "loaded_files":  set(),   # track which file names already parsed
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# Load analysis modules once
if not st.session_state.analyses:
    st.session_state.analyses = discover(ANALYSIS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _windowed_png(arr, ww, wl):
    lo = wl - ww / 2;  hi = wl + ww / 2
    norm = ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
    return Image.fromarray(norm, "L").convert("RGB")

def _default_wl(arr):
    mn, mx = float(np.min(arr)), float(np.max(arr))
    return max(mx - mn, 1.0), (mn + mx) / 2

def _tag(ds, a, d=""):
    try:
        v = getattr(ds, a, None)
        if v is None: return d
        if hasattr(v, "__len__") and not isinstance(v, str):
            v = v[0] if len(v) else d
        return str(v).strip() or d
    except: return d

def _tagf(ds, a):
    try:
        v = getattr(ds, a, None)
        if v is None: return None
        if hasattr(v, "__len__") and not isinstance(v, str): v = v[0]
        return float(v)
    except: return None

def _fd(d): return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d)==8 and d.isdigit() else d
def _ft(t): t=t.split(".")[0]; return f"{t[:2]}:{t[2:4]}:{t[4:6]}" if len(t)>=6 else t

def _annotations(ds, idx, n):
    if ds is None: return {"TL":[],"TR":[],"BL":[],"BR":[]}
    name = str(_tag(ds,"PatientName","Unknown")).replace("^"," ").strip()
    tl = [name]
    for a,l in [("PatientID","ID"),("PatientBirthDate","DOB"),("PatientSex","Sex")]:
        v=_tag(ds,a)
        if v: tl.append(f"{l}: {_fd(v) if 'Date' in a else v}")
    sd=_tag(ds,"StudyDate"); st2=_tag(ds,"StudyTime")
    if sd: tl.append(f"Date: {_fd(sd)}  {_ft(st2)}".rstrip())
    for a in ("StudyDescription","AccessionNumber"):
        v=_tag(ds,a)
        if v: tl.append(v if a=="StudyDescription" else f"Acc: {v}")

    mod=_tag(ds,"Modality","")
    tr=[]
    for a in ("InstitutionName",):
        v=_tag(ds,a)
        if v: tr.append(v)
    mfr=_tag(ds,"Manufacturer"); mdl=_tag(ds,"ManufacturerModelName")
    if mfr or mdl: tr.append(f"{mfr} {mdl}".strip())
    for a,l in [("Modality","Modality"),("SeriesDescription",""),
                ("SeriesNumber","Series"),("ProtocolName","Protocol")]:
        v=_tag(ds,a)
        if v: tr.append(v if not l else f"{l}: {v}")
    if mod in ("CT",""):
        for a,l in [("KVP","kVp"),("XRayTubeCurrent","mA"),("Exposure","mAs"),("CTDIvol","CTDIvol")]:
            v=_tagf(ds,a)
            if v is not None: tr.append(f"{l}: {v:.0f}"+(" mGy" if a=="CTDIvol" else ""))
        for a,l in [("SpiralPitchFactor","Pitch"),("RevolutionTime","Rot time")]:
            v=_tagf(ds,a)
            if v is not None and 0.1<v<10: tr.append(f"{l}: {v:.2f}")
    elif mod=="MR":
        for a,l,u in [("MagneticFieldStrength","Field"," T"),("RepetitionTime","TR"," ms"),
                      ("EchoTime","TE"," ms"),("FlipAngle","Flip","°")]:
            v=_tagf(ds,a)
            if v is not None: tr.append(f"{l}: {v:.1f}{u}")
    elif mod in ("PT","NM"):
        try:
            rp=ds.RadiopharmaceuticalInformationSequence[0]
            r2=str(getattr(rp,"Radiopharmaceutical","")).strip()
            d2=float(getattr(rp,"RadionuclideTotalDose",0))
            if r2: tr.append(f"Tracer: {r2}")
            if d2: tr.append(f"Dose: {d2/1e6:.1f} MBq")
        except: pass

    ps=getattr(ds,"PixelSpacing",None)
    thick=_tagf(ds,"SliceThickness"); loc=_tagf(ds,"SliceLocation")
    bl=[f"Slice: {idx+1} / {n}"]
    if loc   is not None: bl.append(f"Loc: {loc:.2f} mm")
    if thick is not None: bl.append(f"Thickness: {thick:.2f} mm")
    rows=_tag(ds,"Rows"); cols=_tag(ds,"Columns")
    if rows and cols: bl.append(f"Matrix: {rows} × {cols}")
    if ps:
        try: bl.append(f"Pixel: {float(ps[0]):.3f} × {float(ps[1]):.3f} mm")
        except: pass
    if ps and rows and cols:
        try: bl.append(f"FOV: {float(ps[0])*int(rows):.0f} × {float(ps[1])*int(cols):.0f} mm")
        except: pass

    br=[]
    it=_tag(ds,"ImageType")
    if it: br.append(it.replace("\\","/ ")[:30])
    for a,l in [("ConvolutionKernel","Kernel"),("PatientPosition","Position")]:
        v=_tag(ds,a)
        if v: br.append(f"{l}: {v}")
    rd=_tagf(ds,"ReconstructionDiameter")
    if rd: br.append(f"Recon Ø: {rd:.0f} mm")
    return {"TL":tl,"TR":tr,"BL":bl,"BR":br}


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(180deg,#060d1f 0%,#0e2248 55%,#1a3d7a 100%);
  padding:14px 24px 10px;border-bottom:1px solid #1a3a6a;
  margin:-1rem -1rem 1.5rem -1rem;display:flex;align-items:center;gap:16px">
  <div>
    <p style="font-size:24px;font-weight:700;color:#fff;letter-spacing:1px;margin:0">
      ⛰️ CAMP.png</p>
    <p style="font-size:9px;color:rgba(160,200,240,.7);letter-spacing:3px;margin:2px 0 0">
      DICOM VIEWER &amp; ANALYSIS</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown("### 📂 Load DICOM Files")
    st.caption("Select individual .dcm / .ima files, or a .zip of a DICOM folder.")

    # No `type` filter — browsers hide .dcm/.ima if we filter by extension
    uploaded = st.file_uploader(
        "Upload DICOM files",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Load uploaded files", use_container_width=True, type="primary"):
        if not uploaded:
            st.warning("Please select files first using the chooser above.")
        else:
            # Deduplicate by filename so re-clicking doesn't reload
            new_files = [f for f in uploaded
                         if f.name not in st.session_state.loaded_files]
            if not new_files:
                st.info("All selected files are already loaded.")
            else:
                with st.spinner(f"Reading {len(new_files)} file(s)…"):
                    pairs = []
                    for f in new_files:
                        data = f.read()
                        if f.name.lower().endswith(".zip"):
                            try:
                                with zipfile.ZipFile(io.BytesIO(data)) as z:
                                    for zname in z.namelist():
                                        if not zname.endswith("/"):
                                            pairs.append((os.path.basename(zname),
                                                          z.read(zname)))
                            except Exception:
                                pairs.append((f.name, data))
                        else:
                            pairs.append((f.name, data))

                    if not pairs:
                        st.error("No readable files found.")
                    else:
                        try:
                            new_groups = load_bytes_as_series(pairs)
                            added = 0
                            for lbl, series in new_groups.items():
                                if lbl not in st.session_state.series_map:
                                    st.session_state.series_map[lbl] = series
                                    added += 1

                            # Mark files as loaded
                            for f in new_files:
                                st.session_state.loaded_files.add(f.name)

                            if added:
                                st.success(f"✓ Loaded {added} series "
                                           f"({len(pairs)} files)")
                                # Auto-select first if nothing active
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
                            else:
                                st.info("All series already loaded.")
                        except Exception as e:
                            st.error(f"Failed to parse DICOM: {e}")
                            with st.expander("Details"):
                                st.code(traceback.format_exc())

    # ── Series list ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Series")
    series_map = st.session_state.series_map

    if not series_map:
        st.caption("No series loaded yet. Upload files above.")
    else:
        for lbl, series in list(series_map.items()):
            meta   = series["metadata"]
            active = (lbl == st.session_state.active_series)
            c1, c2 = st.columns([5, 1])
            with c1:
                desc  = meta.get("series_desc", lbl) or lbl
                n_sl  = meta.get("n_slices", 0)
                mod   = meta.get("modality", "")
                label_txt = f"{'▶ ' if active else ''}{desc}"
                if st.button(label_txt, key=f"sel_{lbl}",
                             use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.active_series = lbl
                    n = meta["n_slices"]
                    st.session_state.slice_idx = n // 2
                    arr = series["images"][n // 2]
                    ww, wl = _default_wl(arr)
                    st.session_state.ww = ww
                    st.session_state.wl = wl
                    st.rerun()
                st.caption(f"{mod} · {n_sl} slices")
            with c2:
                if st.button("✕", key=f"rm_{lbl}", help="Remove"):
                    del st.session_state.series_map[lbl]
                    if st.session_state.active_series == lbl:
                        rem = list(st.session_state.series_map.keys())
                        st.session_state.active_series = rem[0] if rem else None
                    st.rerun()

        st.markdown("")
        if st.button("✕ Clear all", use_container_width=True):
            st.session_state.series_map     = {}
            st.session_state.active_series  = None
            st.session_state.loaded_files   = set()
            st.rerun()

    # ── Window / Level ───────────────────────────────────────────────────────
    active_lbl = st.session_state.active_series
    if active_lbl and active_lbl in series_map:
        st.markdown("---")
        st.markdown("### 🔆 Window / Level")
        arr_wl = series_map[active_lbl]["images"][st.session_state.slice_idx]
        mn, mx = float(np.min(arr_wl)), float(np.max(arr_wl))
        rng = max(mx - mn, 1.0)
        ww_s = st.slider("Width",  1.0, rng*2,
                          float(st.session_state.ww or rng),
                          step=max(rng/200, 0.1), format="%.0f",
                          key="ww_slider")
        wl_s = st.slider("Level", mn - rng*0.2, mx + rng*0.2,
                          float(st.session_state.wl or (mn+mx)/2),
                          step=max(rng/200, 0.1), format="%.0f",
                          key="wl_slider")
        st.session_state.ww = ww_s
        st.session_state.wl = wl_s
        if st.button("Reset", use_container_width=True):
            ww0, wl0 = _default_wl(arr_wl)
            st.session_state.ww = ww0; st.session_state.wl = wl0
            st.rerun()

    st.markdown("---")
    st.session_state.show_anno = st.toggle(
        "Show DICOM annotations", value=st.session_state.show_anno)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — Viewer + Analysis side by side
# ══════════════════════════════════════════════════════════════════════════════
active_lbl = st.session_state.active_series

# Show upload prompt when nothing loaded
if not active_lbl or active_lbl not in series_map:
    st.info("👈 Upload DICOM files in the sidebar to get started.")
    # Still show analysis modules so users know they exist
    st.markdown("---")
    st.markdown("#### 🔬 Available Analysis Modules")
    for name, desc, _ in st.session_state.analyses:
        st.markdown(f"**{name}** — {desc}")
    st.stop()

# Active series
series   = series_map[active_lbl]
images   = series["images"]
infos    = series["infos"]
metadata = series["metadata"]
n_slices = len(images)

viewer_col, analysis_col = st.columns([3, 2])

# ── Viewer ────────────────────────────────────────────────────────────────────
with viewer_col:
    slice_idx = st.slider(
        f"Slice  ({n_slices} total)", 0, n_slices - 1,
        st.session_state.slice_idx, key="main_slider")
    st.session_state.slice_idx = slice_idx

    arr = images[slice_idx]
    ds  = infos[slice_idx] if slice_idx < len(infos) else None
    ww  = st.session_state.ww or _default_wl(arr)[0]
    wl  = st.session_state.wl or _default_wl(arr)[1]

    img_pil = _windowed_png(arr, ww, wl)
    st.image(img_pil, use_container_width=True,
             caption=f"Slice {slice_idx+1}/{n_slices}  W:{ww:.0f}  L:{wl:.0f}")

    if st.session_state.show_anno and ds is not None:
        anno = _annotations(ds, slice_idx, n_slices)
        a1, a2 = st.columns(2)
        with a1:
            if anno["TL"]:
                st.markdown('<div class="anno-box">' +
                            "\n".join(anno["TL"]) + "</div>",
                            unsafe_allow_html=True)
            if anno["BL"]:
                st.markdown('<div class="anno-box">' +
                            "\n".join(anno["BL"]) + "</div>",
                            unsafe_allow_html=True)
        with a2:
            if anno["TR"]:
                st.markdown('<div class="anno-box">' +
                            "\n".join(anno["TR"]) + "</div>",
                            unsafe_allow_html=True)
            br = [f"W: {ww:.0f}  L: {wl:.0f}"] + anno["BR"]
            st.markdown('<div class="anno-box">' +
                        "\n".join(br) + "</div>",
                        unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
with analysis_col:
    st.markdown("#### 🔬 Analysis")

    analyses = st.session_state.analyses
    if not analyses:
        st.warning("No analysis modules found in the `analysis/` folder.")
    else:
        for name, desc, module in analyses:
            with st.expander(f"**{name}**", expanded=True):
                st.caption(desc)

                needs_ds = (getattr(module, "NEEDS_DATASETS", False)
                            or "acr" in name.lower())
                data = infos if (needs_ds and infos) else images

                if st.button(f"▶ Run", key=f"run_{name}",
                             use_container_width=True, type="primary"):
                    prog = st.progress(0, text="Starting…")
                    try:
                        result = module.run(
                            data, metadata,
                            progress_cb=lambda v, p=prog:
                                p.progress(min(v, 100),
                                           text=f"Running… {v}%"))
                        prog.progress(100, text="Done ✓")
                        st.session_state[f"result_{name}"] = result

                        # Inline summary
                        inter = result.get("interslice")
                        intra = result.get("intraslice")
                        n_src = result.get("n_sources")
                        if inter is not None and not (
                                isinstance(inter, float) and
                                np.isnan(inter)):
                            mc1, mc2 = st.columns(2)
                            mc1.metric("Interslice", f"{inter:.2f}%")
                            mc2.metric("Intraslice", f"{intra:.2f}%")
                        if n_src is not None:
                            st.metric("Sources found", n_src)

                        errs = result.get("errors", {})
                        if isinstance(errs, dict) and errs:
                            for en, et in errs.items():
                                with st.expander(f"⚠ {en} error"):
                                    st.code(et)

                    except Exception:
                        prog.empty()
                        st.error("Analysis failed")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())

                # Cached results
                cached = st.session_state.get(f"result_{name}")
                if cached:
                    # Tables
                    shown = set()
                    for key, lbl in [("dataframe","Results"),
                                     ("df_module1","Module 1"),
                                     ("df_module2","Module 2"),
                                     ("df_module3","Module 3"),
                                     ("df_module4","Module 4")]:
                        df = cached.get(key)
                        if df is not None and not df.empty and id(df) not in shown:
                            shown.add(id(df))
                            st.caption(lbl)
                            st.dataframe(df, use_container_width=True,
                                         hide_index=True)

                    # Figures
                    figs = cached.get("figures", [])
                    if figs:
                        tabs = st.tabs([f"Fig {i+1}" for i in range(len(figs))])
                        for tab, fig in zip(tabs, figs):
                            with tab:
                                st.pyplot(fig, use_container_width=True)

                    # Excel export
                    import pandas as pd
                    dfs = {k: cached[k] for k in
                           ("dataframe","df_module1","df_module2",
                            "df_module3","df_module4")
                           if cached.get(k) is not None
                           and not cached[k].empty}
                    if dfs:
                        buf = io.BytesIO()
                        sheets = {"dataframe":"Results","df_module1":"Module1",
                                  "df_module2":"Module2","df_module3":"Module3",
                                  "df_module4":"Module4"}
                        with pd.ExcelWriter(buf, engine="openpyxl") as w:
                            for k, df in dfs.items():
                                df.to_excel(w, sheet_name=sheets[k], index=False)
                        st.download_button(
                            "⬇ Export Excel",
                            data=buf.getvalue(),
                            file_name=f"{name.replace(' ','_')}_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument"
                                 ".spreadsheetml.sheet",
                            key=f"export_{name}",
                            use_container_width=True,
                        )

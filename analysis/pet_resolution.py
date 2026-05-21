"""PET Spatial Resolution — FWHM of all point sources via 3-D MIP."""
DISPLAY_NAME = "PET Spatial Resolution"
DESCRIPTION  = "FWHM (X/Y/Z) of all point sources via 3-D MIP + Gaussian fitting"
PARAMETERS   = {}

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from skimage.measure import label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _gaussian(x, amp, mean, sd, off):
    return amp * np.exp(-(x - mean)**2 / (2 * sd**2)) + off


def _fit_fwhm(profile, pix_mm):
    profile = np.asarray(profile, dtype=np.float64)
    n = len(profile)
    if n < 4 or np.ptp(profile) == 0:
        return float("nan"), None, None, None
    x = np.arange(n, dtype=np.float64)
    try:
        p0 = [float(np.max(profile) - np.min(profile)),
              float(x[np.argmax(profile)]),
              float((n - 1) / 6.0),
              float(np.min(profile))]
        popt, _ = curve_fit(_gaussian, x, profile, p0=p0, maxfev=8000)
        xf = np.linspace(0, n - 1, 100)
        yf = _gaussian(xf, *popt)
        peaks, _ = find_peaks(yf)
        if not len(peaks): return float("nan"), None, None, None
        bp = [peaks[int(np.argmax(yf[peaks]))]]
        ws = peak_widths(yf, bp, rel_height=0.5)
        sc = (n - 1) / 99.0
        return float(ws[0][0]) * sc * pix_mm, yf, float(ws[2][0]) * sc, float(ws[3][0]) * sc
    except Exception:
        return float("nan"), None, None, None


def run(images_sorted: list, metadata: dict, progress_cb=None) -> dict:
    dx = float(metadata.get("dx", 1.0))
    dy = float(metadata.get("dy", 1.0))
    dz = float(metadata.get("dz", dy))
    if dz <= 0:
        infos = metadata.get("infos", [])
        if infos:
            try: dz = float(getattr(infos[0], "SliceThickness", dy))
            except Exception: dz = dy

    images = np.array(images_sorted, dtype=np.float32)
    nz, nr, nc = images.shape
    if progress_cb: progress_cb(5)

    mip = np.max(images, axis=0)
    gmax = float(np.max(mip))
    if gmax == 0: raise ValueError("All pixels are zero.")
    filled  = binary_fill_holes(mip > gmax * 0.10)
    labeled = label(filled)
    regions = regionprops(labeled)
    if not regions: raise ValueError("No sources detected above threshold.")

    sources = []
    for r in regions:
        cy = int(round(r.centroid[0])); cx = int(round(r.centroid[1]))
        peak_sl = int(np.argmax(images[:, cy, cx]))
        sources.append({"cy": cy, "cx": cx, "peak_slice": peak_sl})
    sources.sort(key=lambda s: s["peak_slice"])
    if progress_cb: progress_cb(15)

    rows_out = []; figs = []; roi_coords = []
    hw = 10; pstep = 80.0 / max(len(sources), 1)

    for i, src in enumerate(sources):
        cy, cx, ps = src["cy"], src["cx"], src["peak_slice"]
        pi = images[ps]
        xp = pi[cy, max(cx-hw,0):min(cx+hw,nc)].tolist()
        yp = pi[max(cy-hw,0):min(cy+hw,nr), cx].tolist()
        zp = images[:, cy, cx].tolist()
        fx, yf_x, xl, xr = _fit_fwhm(xp, dx)
        fy, yf_y, yl, yr = _fit_fwhm(yp, dy)
        fz, yf_z, zl, zr = _fit_fwhm(zp, dz)
        xd = ((nc/2) - cx) * dx * 0.1
        yd = ((nr/2) - cy) * dy * 0.1
        zd = ps * dz * 0.1
        rows_out.append({"Source": i+1, "Peak Slice": ps+1,
                         "X pos (cm)": round(xd,2), "Y pos (cm)": round(yd,2),
                         "Z pos (cm)": round(zd,2),
                         "FWHM_X (mm)": round(fx,3) if not np.isnan(fx) else None,
                         "FWHM_Y (mm)": round(fy,3) if not np.isnan(fy) else None,
                         "FWHM_Z (mm)": round(fz,3) if not np.isnan(fz) else None})
        roi_coords.append({"slice_idx": ps, "radius_px": 12,
                           "center": [cy, cx],
                           "top": None, "bottom": None, "left": None, "right": None})

        fig, axs = plt.subplots(1, 4, figsize=(14, 3), facecolor="white")
        for ax in axs: ax.set_facecolor("white")
        fig.suptitle(f"Source {i+1}  peak sl {ps+1}  ({xd:.2f},{yd:.2f},{zd:.2f} cm)", fontsize=9)
        axs[0].imshow(mip, cmap="hot"); axs[0].plot(cx, cy, "c+", ms=10, mew=1.5)
        axs[0].set_title("MIP", fontsize=8); axs[0].axis("off")
        for ax, (prof, yf, l2, r2, lbl2, pm, fwv) in zip(axs[1:], [
            (xp, yf_x, xl, xr, "X", dx, fx),
            (yp, yf_y, yl, yr, "Y", dy, fy),
            (zp, yf_z, zl, zr, "Z", dz, fz)]):
            ax.plot(np.arange(len(prof))*pm, prof, "o", ms=3, color="steelblue", label="data")
            if yf is not None:
                ax.plot(np.linspace(0,(len(prof)-1)*pm,100), yf, color="tomato", lw=1.5, label="fit")
                if l2 is not None: ax.axvspan(l2*pm, r2*pm, alpha=.15, color="green")
            ax.set_xlabel(f"{lbl2} (mm)"); ax.set_ylabel("Signal")
            fstr = f"{fwv:.2f} mm" if not np.isnan(fwv) else "N/A"
            ax.set_title(f"FWHM={fstr}", fontsize=8); ax.legend(fontsize=7)
        plt.tight_layout(); figs.append(fig)
        if progress_cb: progress_cb(int(15 + (i+1)*pstep))

    if progress_cb: progress_cb(100)
    return {"dataframe": pd.DataFrame(rows_out), "figures": figs,
            "roi_coords": roi_coords,
            "slices_used": sorted({s["peak_slice"]+1 for s in sources}),
            "n_sources": len(sources)}

"""PET Uniformity — 5-ROI interslice/intraslice analysis."""
DISPLAY_NAME = "PET Uniformity"
DESCRIPTION  = "Interslice & intraslice uniformity via 5-ROI method (EARL/NEMA)"
PARAMETERS   = {}

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def _circle_mask(cy, cx, radius, rows, cols):
    ys, xs = np.ogrid[:rows, :cols]
    return ((xs - cx)**2 + (ys - cy)**2) <= radius**2


def run(images_sorted: list, metadata: dict, progress_cb=None) -> dict:
    dx   = metadata.get("dx", 1.0)
    dy   = metadata.get("dy", 1.0)
    rows = metadata.get("rows", images_sorted[0].shape[0])
    cols = metadata.get("cols", images_sorted[0].shape[1])

    slices_used  = []
    center_rois  = []; top_rois    = []; bottom_rois = []
    left_rois    = []; right_rois  = []
    roi_coords   = []; errors      = []
    n = len(images_sorted)

    for k, img in enumerate(images_sorted):
        if progress_cb: progress_cb(int(k / n * 90))
        if np.max(img) <= 0: continue
        try:
            thresh = (img > np.max(img) * 0.10).astype(np.uint8)
            mask   = clear_border(thresh)
            mask   = ndi.binary_fill_holes(mask).astype(np.uint8)
            labeled = label(mask)
            if labeled.max() == 0: errors.append(k + 1); continue
            rps  = regionprops(labeled)
            best = rps[int(np.argmax([r.area for r in rps]))]
            filled = np.zeros_like(labeled)
            filled[tuple(best.coords.T)] = 1
            regions  = regionprops(label(filled))
            centroid = [int(round(v)) for v in regions[0].centroid]
            diameter = min(int(round(regions[0].axis_major_length)),
                          int(round(regions[0].axis_minor_length)))
            r_px   = int(round(15.0 / dx))
            offset = int(round(25.0 / dy))
            top_e  = [int(round(centroid[0] - diameter / 2)), centroid[1]]
            bot_e  = [int(round(centroid[0] + diameter / 2)), centroid[1]]
            lft_e  = [centroid[0], int(round(centroid[1] - diameter / 2))]
            rgt_e  = [centroid[0], int(round(centroid[1] + diameter / 2))]
            centres = {
                "top"   : [top_e[0] + offset,   centroid[1]],
                "bottom": [bot_e[0] - offset,   centroid[1]],
                "left"  : [centroid[0], lft_e[1] + offset],
                "right" : [centroid[0], rgt_e[1] - offset],
                "center": [centroid[0], centroid[1]],
            }
            def sample(cy2, cx2):
                m = _circle_mask(cy2, cx2, r_px, rows, cols)
                pts = img[m]
                return float(np.mean(pts)) if len(pts) > 0 else float("nan")
            vals = {kk: sample(centres[kk][0], centres[kk][1]) for kk in centres}
            slices_used.append(k + 1)
            center_rois.append(vals["center"]); top_rois.append(vals["top"])
            bottom_rois.append(vals["bottom"]); left_rois.append(vals["left"])
            right_rois.append(vals["right"])
            roi_coords.append({"slice_idx": k, "radius_px": r_px, **centres})
        except Exception: errors.append(k + 1)

    if progress_cb: progress_cb(95)
    df = pd.DataFrame({"Slice": slices_used, "Center": center_rois,
                       "Top": top_rois, "Bottom": bottom_rois,
                       "Left": left_rois, "Right": right_rois})
    roi_cols = ["Center", "Top", "Bottom", "Left", "Right"]

    def interslice(df):
        if df.empty: return float("nan")
        sm = df[roi_cols].mean(axis=1)
        return float(100.0 * (sm.max() - sm.min()) / sm.mean()) if sm.mean() != 0 else float("nan")

    def intraslice(df):
        if df.empty: return float("nan")
        ps = df[roi_cols].apply(
            lambda r: 100.0 * (r.max() - r.min()) / r.mean() if r.mean() != 0 else float("nan"), axis=1)
        return float(ps.mean())

    if progress_cb: progress_cb(100)
    return {"dataframe": df, "interslice": interslice(df), "intraslice": intraslice(df),
            "roi_coords": roi_coords, "slices_used": slices_used, "errors": errors}

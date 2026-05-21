import os
import numpy as np
import pydicom

SUPPORTED_EXTENSIONS = {".dcm", ".ima"}


def _is_dicom(fpath: str) -> bool:
    ext = os.path.splitext(fpath)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
        return True
    if ext == "":
        try:
            with open(fpath, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except Exception:
            return False
    return False


def _collect_candidates(root: str) -> list:
    candidates = []
    seen_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen_dirs:
            dirnames.clear()
            continue
        seen_dirs.add(real)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if _is_dicom(fpath):
                candidates.append(fpath)
    return candidates


def _series_key(ds) -> tuple:
    desc = (str(getattr(ds, "SeriesDescription", "")).strip()
            or str(getattr(ds, "SeriesNumber", "")).strip()
            or "Unknown")
    time = str(getattr(ds, "SeriesTime", "000000")).split(".")[0].zfill(6)
    try:
        num = int(getattr(ds, "SeriesNumber", 0))
    except Exception:
        num = 0
    return (desc, time, num)


def load_folder_as_series(folder_path: str) -> dict:
    candidates = _collect_candidates(folder_path)
    if not candidates:
        raise ValueError(f"No DICOM files found in: {folder_path}")

    groups: dict = {}
    failed = 0
    for fpath in candidates:
        try:
            ds = pydicom.dcmread(fpath)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            img = ds.pixel_array.astype(np.float32) * slope + intercept
            instance = int(getattr(ds, "InstanceNumber", 0))
            key = _series_key(ds)
        except Exception:
            failed += 1
            continue
        groups.setdefault(key, []).append((instance, img, ds))

    if not groups:
        raise ValueError(f"Could not decode pixel data from any DICOM file ({failed} failed).")

    result = {}
    seen_labels = set()
    for key in sorted(groups.keys(), key=lambda k: (k[1], k[2])):
        entries = sorted(groups[key], key=lambda e: e[0])
        images = [e[1] for e in entries]
        infos  = [e[2] for e in entries]
        ds0 = infos[0]
        ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        t = key[1]
        time_str = f"  {t[:2]}:{t[2:4]}:{t[4:6]}" if t != "000000" else ""
        base_label = f"{key[0]}{time_str}  ({len(images)} sl)"
        label = base_label
        sfx = 1
        while label in seen_labels:
            sfx += 1
            label = f"{base_label} [{sfx}]"
        seen_labels.add(label)
        result[label] = {
            "images": images,
            "infos":  infos,
            "metadata": {
                "dx":          float(ps[1]),
                "dy":          float(ps[0]),
                "rows":        int(getattr(ds0, "Rows",    images[0].shape[0])),
                "cols":        int(getattr(ds0, "Columns", images[0].shape[1])),
                "modality":    str(getattr(ds0, "Modality",    "?")),
                "patient_id":  str(getattr(ds0, "PatientID",   "?")),
                "study_date":  str(getattr(ds0, "StudyDate",   "")),
                "series_desc": key[0],
                "series_time": key[1],
                "n_slices":    len(images),
                "folder":      folder_path,
                "infos":       infos,
            },
        }
    return result


def load_bytes_as_series(file_bytes_list: list) -> dict:
    """Load from list of (filename, bytes) — used for uploaded files."""
    import io, tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        for fname, data in file_bytes_list:
            safe = os.path.basename(fname).replace("..", "_")
            dest = os.path.join(tmp, safe)
            with open(dest, "wb") as f:
                f.write(data)
        return load_folder_as_series(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

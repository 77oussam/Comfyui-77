# curve77.py  — single-file robust Curve node
# Works alone (no curve77_safe/unified). Handles many curve formats safely.

import numpy as np

# --- Torch bridging (handles ComfyUI IMAGE tensors) -------------------------
try:
    import torch
except Exception:
    torch = None
import numpy as np

def _to_numpy_batch(img):
    """
    Accept torch tensor [B,H,W,C] or [H,W,C] (0..1 float),
    or numpy array. Return (np_uint8[B,H,W,C], was_torch, like_tensor).
    """
    if torch is not None and isinstance(img, torch.Tensor):
        like = img
        t = img
        if t.dim() == 3:  # [H,W,C] -> add batch
            t = t.unsqueeze(0)
        # clamp to 0..1 then to uint8 on CPU
        arr = (t.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        return arr, True, like
    # numpy path
    arr = np.asarray(img)
    if arr.ndim == 3:      # [H,W,C] -> add batch
        arr = arr[None, ...]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr, False, None

def _from_numpy_batch(arr_u8, was_torch, like):
    """
    Convert uint8 numpy [B,H,W,C] back to original container as 0..1 float.
    """
    arr_f32 = arr_u8.astype(np.float32) / 255.0
    if was_torch and torch is not None:
        t = torch.from_numpy(arr_f32)
        if like.dim() == 3:  # original had no batch
            t = t.squeeze(0)
        return t
    # numpy case
    if arr_f32.shape[0] == 1:
        arr_f32 = arr_f32[0]
    return arr_f32

NODE_NAME = "Curve 77"
CATEGORY = "ALL-77/Color"

def _identity_points():
    return [[0.0, 0.0], [1.0, 1.0]]

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

def _try_to_list(data):
    # detach torch / numpy gracefully if present
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
    except Exception:
        pass
    try:
        import numpy as np  # noqa
        if isinstance(data, np.ndarray):
            return data.tolist()
    except Exception:
        pass
    return data

def _norm_pair(x, y):
    # make (x,y) numeric in [0,1]; autoscale from 0-255 if needed
    xf = float(x)
    yf = float(y)
    if xf > 1.0 or yf > 1.0:
        xf /= 255.0
        yf /= 255.0
    return [_clamp01(xf), _clamp01(yf)]

def _as_points_list(data):
    """
    Accept:
      [[x,y],...], [{'x':x,'y':y},...], {'points':...}, {'curve':...}, {'data':...},
      {'x':[...],'y':[...]}, numeric-key dict {'0':[x,y],...} or {'0':{'x':..,'y':..}},
      flat [x0,y0,x1,y1,...], numpy/torch, None/True/False → identity.
    Return normalized [[x,y], ...] in [0,1].
    """
    if data in (None, True, False):
        return _identity_points()

    data = _try_to_list(data)

    # dict shapes
    if isinstance(data, dict):
        for key in ("points", "curve", "data"):
            if key in data:
                return _as_points_list(data[key])
        if "x" in data and "y" in data:
            xs = _try_to_list(data["x"])
            ys = _try_to_list(data["y"])
            pts = []
            try:
                for x, y in zip(xs, ys):
                    pts.append(_norm_pair(x, y))
                return pts if pts else _identity_points()
            except Exception:
                return _identity_points()

        # numeric keys
        try:
            keys = list(data.keys())
            try:
                keys = sorted(keys, key=lambda k: int(k))
            except Exception:
                keys = sorted(keys)
            vals = []
            for k in keys:
                v = data[k]
                if isinstance(v, dict) and "x" in v and "y" in v:
                    vals.append([v["x"], v["y"]])
                else:
                    vals.append(v)
            return _as_points_list(vals)
        except Exception:
            return _identity_points()

    # list / tuple
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return _identity_points()

        # list of dicts [{'x':..,'y':..}, ...]
        if isinstance(data[0], dict) and "x" in data[0] and "y" in data[0]:
            pts = []
            try:
                for p in data:
                    pts.append(_norm_pair(p["x"], p["y"]))
                return pts if pts else _identity_points()
            except Exception:
                return _identity_points()

        # [[x,y], ...]
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in data):
            pts = []
            try:
                for x, y in data:
                    pts.append(_norm_pair(x, y))
                return pts if pts else _identity_points()
            except Exception:
                return _identity_points()

        # [xs, ys]
        if len(data) == 2 and all(isinstance(v, (list, tuple)) for v in data):
            xs, ys = data
            pts = []
            try:
                for x, y in zip(xs, ys):
                    pts.append(_norm_pair(x, y))
                return pts if pts else _identity_points()
            except Exception:
                return _identity_points()

        # flat [x0,y0,x1,y1,...]
        ok = True
        for v in data:
            try:
                float(v)
            except Exception:
                ok = False
                break
        if ok and len(data) % 2 == 0:
            it = iter(data)
            pts = []
            try:
                for x, y in zip(it, it):
                    pts.append(_norm_pair(x, y))
                return pts if pts else _identity_points()
            except Exception:
                return _identity_points()

    # anything else
    return _identity_points()

def _complete_and_sort(points, add_endpoints=True):
    """Sort by x, dedupe, optionally ensure [0,0] and [1,1]."""
    pts = sorted(points, key=lambda p: p[0])
    # remove duplicates by x keeping last
    dedup = {}
    for x, y in pts:
        dedup[float(x)] = float(y)
    pts = [[x, dedup[x]] for x in sorted(dedup.keys())]
    if add_endpoints:
        if not pts or pts[0][0] > 0.0:
            pts = [[0.0, 0.0]] + pts
        if pts[-1][0] < 1.0:
            pts = pts + [[1.0, 1.0]]
    return pts

def _lut_from_points(points):
    """Build 256-entry LUT using linear interpolation over control points."""
    pts = _complete_and_sort(points)
    xs = np.array([p[0] for p in pts], dtype=np.float32) * 255.0
    ys = np.array([p[1] for p in pts], dtype=np.float32) * 255.0
    xgrid = np.arange(256, dtype=np.float32)
    lut = np.interp(xgrid, xs, ys).clip(0, 255).astype(np.uint8)
    return lut

def _apply_lut_to_image(image, lut, channel="RGB"):
    """
    Supports torch tensors or numpy arrays.
    Works with [B,H,W,C] or [H,W,C], C>=1, values 0..1.
    """
    batch_u8, was_torch, like = _to_numpy_batch(image)   # -> uint8 [B,H,W,C]
    B, H, W, C = batch_u8.shape

    out = batch_u8.copy()

    if C == 1:
        out[..., 0] = lut[out[..., 0]]
    else:
        if channel == "RGB":
            for c in range(min(3, C)):
                out[..., c] = lut[out[..., c]]
        else:
            idx = {"Red": 0, "Green": 1, "Blue": 2}.get(channel, 0)
            if idx < C:
                out[..., idx] = lut[out[..., idx]]

    return _from_numpy_batch(out, was_torch, like)

class Curve77:
    """
    One node: input image, optional tone_curve socket (works with Curve (mtb) or any format),
    choose channel & mode. In manual mode, if the embedded UI stores points in self.ui_points,
    they’ll be used; otherwise we fall back to tone_curve; if none → identity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["RGB", "Red", "Green", "Blue"],),
                "mode": (["manual", "auto"],),
            },
            "optional": {
                # keep for compatibility with Curve(mtb) etc.
                "tone_curve": ("FLOAT_CURVE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, image, channel="RGB", mode="manual", tone_curve=None):
        print(f"[Curve77] apply called | mode= {mode} | channel= {channel}")
        # pick points source
        points_src = None
        if mode == "manual":
            # if your embedded editor stores points here, we’ll use them
            points_src = getattr(self, "ui_points", None)
        if points_src is None:
            points_src = tone_curve

        # parse robustly; never crash
        try:
            points = _as_points_list(points_src)
        except Exception:
            # last-ditch safety
            points = _identity_points()

        lut = _lut_from_points(points)
        out = _apply_lut_to_image(image, lut, channel=channel)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "Curve77": Curve77,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Curve77": NODE_NAME,
}

"""Document crop augmentations: print defects, overlays, wear, paper, geometry, and scan.
Expects data["image"] as BGR uint8 and returns the updated data.
Textures are stored beside this module; clean paper textures are in paper/.
"""
import os
import random
import stat
from functools import lru_cache

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_GRAIN_DIR = os.path.join(HERE, "paper")
LINES_PER_PAGE = 55          # Approximate crop height as 1/55 of an A4 page height.
SCALE_JITTER = (0.7, 1.4)    # Variation in crop size relative to the texture page.
SIGMA_FRAC = 25 / 2339       # Background blur sigma as a fraction of texture page height.
MIN_H = 13                   # Minimum downsampled crop height, in pixels.
JPEG_Q = (1, 70)             # JPEG quality range; lower values produce stronger compression artifacts.
CURL_AMP = (0, 1)            # Curl displacement in units of crop height.
CURL_LEN = (1, 13)           # Curl extent in units of crop height.
CURL_POWER = (1, 3.5)        # Curl exponent: 1 gives a linear slope; larger values steepen the edge.
CURL_SQUEEZE = (0, 0.8)    # Maximum horizontal compression at the curled edge.
MARKS_FILE = "print.png"
MARKS_BLACK_FILE = "paint_black.png"
SCAN_BLUR = (0.0, 1.0)                # Gaussian blur sigma in current crop pixels.
SCAN_MOTION = (0.0, 6.0)              # Motion trail length in current crop pixels.
SCAN_NOISE = (1.0, 5.0)               # Noise standard deviation on the 0-255 intensity scale.
SCAN_CONTRAST = (1.0, 1.5)            # Contrast multiplier around intensity 127.5.
SCAN_SHIFT = 0.023
MARKS_K = (1, 6)                       # Overlay density: 1 preserves source opacity; larger values increase it.
INK_HUE = (180, 330)                   # Ink hue in degrees, from cyan through blue to magenta.
INK_V = 181


def _to_lin_formula(x):
    x = np.asarray(x, dtype=np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


_LINEAR_UINT8 = _to_lin_formula(np.arange(256, dtype=np.uint8))
_LINEAR_UINT8.flags.writeable = False


def to_lin(x):
    """Convert sRGB intensities from 0-255 to linear light in 0-1."""
    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return _LINEAR_UINT8[x]
    return _to_lin_formula(x)


def to_srgb(y):
    """Convert linear light in 0-1 to sRGB intensities in 0-255."""
    y = np.clip(y, 0, 1)
    return 255.0 * np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


def hue_bgr(hue):
    """Return a BGR color for hue in degrees, full saturation, and value INK_V."""
    hsv = np.uint8([[[round(hue / 2) % 180, 255, INK_V]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def tex_files(tex_dir=HERE):
    """Find JPEG paper/wear textures; root PNG files hold annotation overlays."""
    return sorted(f for f in os.listdir(tex_dir) if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith("_tmp_"))


def read_gray(path):
    t = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if t.shape[1] > t.shape[0]:
        t = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
    return t


@lru_cache(maxsize=1)
def _paper_texture_names(directory, directory_mtime_ns):
    return tuple(sorted(
        (f for f in os.listdir(directory) if f.lower().endswith(".png") and f[:-4].isdigit()),
        key=lambda f: int(f[:-4]),
    ))


@lru_cache(maxsize=8)
def paper_grain_source(path):
    """Load and cache paper textures on demand."""
    return read_gray(path)


@lru_cache(maxsize=1)
def native_texture(path):
    return read_gray(path)


@lru_cache(maxsize=1)
def native_mask(path, mask_path):
    if os.path.isfile(mask_path):
        return read_gray(mask_path)
    return trace_mask(native_texture(path))


def trace_mask(g):
    """Build a uint8 wear mask from dark deviations from the local background."""
    gf = g.astype(np.float32)
    H, W = g.shape
    s = cv2.resize(gf, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(s, (0, 0), SIGMA_FRAC * s.shape[0]), (W, H), interpolation=cv2.INTER_LINEAR)
    return (np.clip((bg - gf) / np.maximum(bg, 1), 0, 1) * 255).astype(np.uint8)


class TexAug:
    def __init__(self, tex_dir=HERE, prob=0.4, curl_prob=0.34, low_prob=0.2, jpeg_prob=0.3,
                 scan_prob=0.3, original_dir=None, marks_prob=None, erase_prob=None,
                 paper_prob=None, print_prob=0.0, **kwargs):
        self.prob = prob
        self.marks_prob = prob if marks_prob is None else marks_prob
        self.erase_prob = prob if erase_prob is None else erase_prob
        self.paper_prob = prob if paper_prob is None else paper_prob
        self.print_prob = print_prob
        self.low_prob = low_prob
        self.scan_prob = scan_prob
        self.jpeg_prob = jpeg_prob
        self.curl_prob = curl_prob
        marks_dir = original_dir if original_dir and all(os.path.isfile(os.path.join(original_dir, f))
                    for f in (MARKS_FILE, MARKS_BLACK_FILE)) else tex_dir
        self.marks_native, self.marks_black = (self._read_rgba(os.path.join(marks_dir, f)) for f in (MARKS_FILE, MARKS_BLACK_FILE))
        self.pages, self.masks, self.names = [], [], []
        self.last = None
        for f in tex_files(tex_dir):
            self.names.append(f)
            original = os.path.join(original_dir, f) if original_dir else None
            if original and os.path.isfile(original):
                self.pages.append(original)
                self.masks.append(os.path.join(original_dir, "masks", os.path.splitext(f)[0] + ".png"))
            else:
                t = read_gray(os.path.join(tex_dir, f))
                self.pages.append(t)
                self.masks.append(trace_mask(t))
        if len(self.pages) < 2:
            raise ValueError(f"в {tex_dir} нужно минимум 2 текстуры, найдено {len(self.pages)}")

    def _page(self, i):
        return native_texture(self.pages[i]) if isinstance(self.pages[i], str) else self.pages[i]

    def _mask(self, i):
        return native_mask(self.pages[i], self.masks[i]) if isinstance(self.masks[i], str) else self.masks[i]

    @staticmethod
    def _read_rgba(path):
        mk = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if mk is None or mk.ndim != 3 or mk.shape[2] != 4:
            raise ValueError(f"{path}: нужен RGBA")
        return cv2.rotate(mk, cv2.ROTATE_90_CLOCKWISE) if mk.shape[1] > mk.shape[0] else mk

    @staticmethod
    def _patch(src, h, w, info=None):
        """Sample a texture region and resize it to an h-by-w float32 patch."""
        return TexAug._fit(TexAug._crop(src, h, w, info), h, w)

    @staticmethod
    def _fit(p, h, w):
        interp = cv2.INTER_AREA if p.shape[0] >= h else cv2.INTER_LINEAR
        return cv2.resize(p, (w, h), interpolation=interp).astype(np.float32)

    @staticmethod
    def _crop(src, h, w, info=None):
        """Sample a full-resolution texture region scaled to an h-by-w crop."""
        H, W = src.shape[:2]
        j = random.uniform(*SCALE_JITTER)
        k = H / (h * LINES_PER_PAGE * j)
        ph, pw = max(1, round(h * k)), max(1, round(w * k))
        if pw > W or ph > H:
            s = min(W / pw, H / ph)
            ph, pw = max(1, int(ph * s)), max(1, int(pw * s))
        y = random.randint(0, H - ph)
        x = random.randint(0, W - pw)
        if info is not None:
            info.update(scale=round(j, 2), x=x, y=y, w=pw, h=ph)
        return src[y:y + ph, x:x + pw]

    def __call__(self, data):
        self.last = {}
        h, w = data["image"].shape[:2]
        if h >= 2 and w >= 2:
            if self.print_prob > 0 and random.random() < self.print_prob:
                data = self._print(data)
            if random.random() < self.marks_prob:
                data = self._marks(data)
            if random.random() < self.erase_prob:
                data = self._erase(data)
            if random.random() < self.paper_prob:
                data = self._paper(data)
            if random.random() < self.curl_prob:
                data = self._curl(data)
            scan = random.random() < self.scan_prob
            if scan:
                data = self._scan(data)
            if random.random() < self.low_prob:
                data = self._lowres(data)
            if scan:
                data = self._noise(data)
            if random.random() < self.jpeg_prob:
                data = self._jpeg(data)
        self.last = self.last or None
        return data

    def _print(self, data):
        """Apply print defects before overlays, wear, geometry, and scanning."""
        ink = random.triangular(0.0, 2.0, 2.0)
        edge = random.triangular(0.0, 1.0, 1.0)
        paper = random.triangular(0.0, 2.0, 2.0)
        seed = random.getrandbits(32)
        data["image"] = self.print_effect(data["image"], ink, edge, paper, seed)
        self.last["печать v2"] = f"тонер {ink:.2f}, край {edge:.2f}, бумага {paper:.2f}, seed {seed}"
        return data

    def _marks(self, data):
        """Multiply linear-light intensities by an RGBA overlay transmission layer."""
        img = data["image"]
        h, w = img.shape[:2]
        info = {}
        r = random.random()
        p = self._crop(self.marks_native if r < 0.8 else self.marks_black, h, w, info)
        k = random.uniform(*MARKS_K)
        a = 1.0 - (1.0 - p[:, :, 3:4].astype(np.float32) / 255.0) ** k
        info["плотность"] = round(k, 2)
        if r < 0.8:
            kind = "родной"
            T = self._fit((1.0 - a * (1.0 - to_lin(p[:, :, :3]))).astype(np.float32), h, w)
        else:
            if r < 0.9:
                kind, c = "чёрный", np.zeros(3, np.float32)
            else:
                hue = random.uniform(*INK_HUE)
                kind, c = f"цветной {hue:.0f}°", to_lin(hue_bgr(hue))
            a = self._fit(a[:, :, 0], h, w)[:, :, None]
            i, e = random.randrange(len(self.masks)), random.random()
            M = self._patch(self._mask(i), h, w)[:, :, None] / 255.0
            a = a * (1.0 - M * e)
            kind += f", стёрты {self.names[i]} e={e:.2f}"
            T = (1.0 - a * (1.0 - c)).astype(np.float32)
        if img.ndim == 2:
            out = to_srgb(to_lin(img) * T.mean(axis=2))
        else:
            out = to_srgb(to_lin(img) * T)
        data["image"] = np.clip(out, 0, 255).astype(np.uint8)
        self.last.update({"пометки": kind, "место_пометок": info})
        return data

    @staticmethod
    def paper_texture_name(seed):
        try:
            directory_stat = os.stat(PAPER_GRAIN_DIR)
        except (FileNotFoundError, NotADirectoryError):
            return None
        if not stat.S_ISDIR(directory_stat.st_mode):
            return None
        names = _paper_texture_names(PAPER_GRAIN_DIR, directory_stat.st_mtime_ns)
        return names[int(seed) % len(names)] if names else None

    @staticmethod
    def _print_paper_patch(h, w, seed):
        name = TexAug.paper_texture_name(seed)
        if not name:
            return None
        texture = paper_grain_source(os.path.join(PAPER_GRAIN_DIR, name))
        H, W = texture.shape
        rng = np.random.default_rng(int(seed) ^ 0x5041504552)
        ph = max(2, round(H / (LINES_PER_PAGE * rng.uniform(*SCALE_JITTER))))
        pw = max(2, round(ph * w / h))
        fit = min(1.0, H / ph, W / pw)
        ph, pw = max(2, int(ph * fit)), max(2, int(pw * fit))
        y, x = int(rng.integers(H - ph + 1)), int(rng.integers(W - pw + 1))
        return cv2.resize(texture[y:y + ph, x:x + pw], (w, h),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def print_effect(img, ink=0.0, edge=0.0, paper=0.0, seed=0):
        """Simulate subpixel boundary defects and toner particles on a 3x grid.

        Defects: Kanungo & Zheng (2004); toner: Norris & Barney Smith (2004).
        Parameters are manually tuned, not calibrated to a specific printer.
        """
        ink, edge, paper = float(ink), float(edge), float(paper)
        if max(ink, edge, paper) <= 0 or min(img.shape[:2]) < 3:
            return img
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light, dark = gray > threshold, gray <= threshold
        if not light.any() or not dark.any():
            return img
        bg = float(np.median(gray[light]))
        fg = float(np.percentile(gray[dark], 10))
        span = bg - fg
        if span < 12:
            return img
        coverage = np.clip((bg - gray.astype(np.float32)) / span, 0, 1)
        binary = (coverage >= 0.5).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if len(stats) > 1:
            components = stats[1:]
            heights = components[:, cv2.CC_STAT_HEIGHT]
            suitable = ((heights >= 5) & (heights < h) &
                        (components[:, cv2.CC_STAT_WIDTH] <= 3 * heights) &
                        (components[:, cv2.CC_STAT_AREA] >= 8))
            letter_h = float(np.median(heights[suitable])) if suitable.any() else h * 0.65
        else:
            letter_h = h * 0.65
        scale = max(0.25, letter_h / 28.0)
        src = img.astype(np.float32)
        out = src
        patch = TexAug._print_paper_patch(h, w, seed)
        if ink > 0 or edge > 0:
            up = 3
            min_sigma = 0.25 * up / 4.0
            size = (w * up, h * up)
            high = cv2.resize(coverage, size, interpolation=cv2.INTER_CUBIC)
            ideal = (high >= 0.5).astype(np.uint8)
            shaped = ideal.copy()
            if edge > 0:
                inside = cv2.distanceTransform(ideal, cv2.DIST_L2, 5)
                outside = cv2.distanceTransform(1 - ideal, cv2.DIST_L2, 5)
                for distance in (inside, outside):
                    np.subtract(distance, 0.5, out=distance)
                    np.maximum(distance, 0, out=distance)
                    np.divide(distance, up, out=distance)
                np.add(inside, outside, out=inside)
                reach = max(0.08, 0.6 * scale * edge)
                np.divide(inside, reach, out=inside)
                np.square(inside, out=inside)
                np.negative(inside, out=inside)
                np.exp(inside, out=inside)
                edge_rng = np.random.default_rng(int(seed) ^ 0x45444745)
                u = edge_rng.random(ideal.shape, dtype=np.float32)
                np.multiply(inside, 0.8 * edge, out=outside)
                shaped[(ideal != 0) & (u < outside)] = 0
                np.multiply(inside, 0.7 * edge, out=outside)
                shaped[(ideal == 0) & (u < outside)] = 1
                shaped = cv2.morphologyEx(shaped, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            printed = shaped.astype(np.float32)
            if ink > 0:
                toner_rng = np.random.default_rng(int(seed) ^ 0x544F4E4552)
                if patch is not None:
                    fiber = patch - cv2.GaussianBlur(patch, (0, 0), max(1.0, 3.0 * scale))
                    fiber -= float(fiber.mean())
                    fiber /= max(float(fiber.std()), 1e-6)
                else:
                    fiber = toner_rng.standard_normal((h, w)).astype(np.float32)
                    fiber = cv2.GaussianBlur(fiber, (0, 0), max(0.3, 0.6 * scale))
                    fiber /= max(float(fiber.std()), 1e-6)
                np.clip(fiber, -3, 3, out=fiber)
                fiber = cv2.resize(fiber, size, interpolation=cv2.INTER_LINEAR)
                radius = max(1, round(0.18 * scale * up))
                particle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (2 * radius + 1, 2 * radius + 1))
                uncovered = min(0.15, max(0.001, 0.045 * ink))
                density = -np.log(uncovered) / float(particle.sum())
                exposure = cv2.GaussianBlur(printed, (0, 0), max(min_sigma, 0.12 * scale * up))
                np.multiply(fiber, -0.16 * ink, out=fiber)
                np.exp(fiber, out=fiber)
                local_density = density * exposure
                np.multiply(local_density, fiber, out=local_density)
                np.negative(local_density, out=local_density)
                np.expm1(local_density, out=local_density)
                np.negative(local_density, out=local_density)
                center_probability = local_density
                centers = (toner_rng.random(ideal.shape, dtype=np.float32) < center_probability).astype(np.uint8)
                printed = cv2.dilate(centers, particle).astype(np.float32)
            optical_sigma = max(min_sigma, 0.18 * scale * up)
            delta = np.subtract(ideal, printed, dtype=np.float32)
            change = cv2.resize(cv2.GaussianBlur(delta, (0, 0), optical_sigma),
                                (w, h), interpolation=cv2.INTER_AREA)
            stats_src = src
            if (img.ndim == 3 and img.shape[2] == 3
                    and np.array_equal(img[:, :, 0], img[:, :, 1])
                    and np.array_equal(img[:, :, 0], img[:, :, 2])):
                stats_src = src[:, :, :1]
            bg_color = np.median(stats_src[light], axis=0)
            fg_color = np.percentile(stats_src[dark], 10, axis=0)
            color_span = np.maximum(bg_color - fg_color, 0)
            out += change * color_span if img.ndim == 2 else change[:, :, None] * color_span
        if paper > 0 and patch is not None:
            transmission = np.maximum(to_lin(patch), 1e-6) ** paper
            out = to_srgb(to_lin(np.clip(out, 0, 255)) *
                          (transmission if img.ndim == 2 else transmission[:, :, None]))
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    @staticmethod
    def scan_shifts(amount, axis=1, sign=1):
        """Place R and B on opposite sides of G along the scan axis."""
        d = amount * sign
        r = (d, 0.0) if axis == 0 else (0.0, d)
        return r, (-r[0], -r[1])

    @staticmethod
    def scan_effect(img, sigma, r_shift=(0.0, 0.0), b_shift=(0.0, 0.0)):
        """Apply blur and RGB offsets; preserve neutral output for grayscale input."""
        shifted = any(r_shift) or any(b_shift)
        mono = shifted and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3
                and np.max(img.max(axis=2) - img.min(axis=2)) <= 2))
        if sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if shifted and img.ndim == 3 and img.shape[2] == 3:
            h, w = img.shape[:2]
            ch = list(cv2.split(img))
            for i, (dx, dy) in ((2, r_shift), (0, b_shift)):
                if dx or dy:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    ch[i] = cv2.warpAffine(ch[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.merge(ch)
            if mono and (any(r_shift) or any(b_shift)):
                img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def motion_effect(img, length, angle):
        """Simulate uniform motion; trail length is in pixels, angle in degrees."""
        if length <= 0:
            return img
        radius = int(np.ceil(length / 2)) + 1
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        t = np.linspace(-length / 2, length / 2, max(3, int(np.ceil(length * 8)) + 1))
        a = np.deg2rad(angle)
        x, y = radius + t * np.cos(a), radius + t * np.sin(a)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - ix, y - iy
        weights = np.ones(len(t), dtype=np.float32)
        weights[[0, -1]] = 0.5
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                np.add.at(kernel, (iy + dy, ix + dx), weights * wx * wy)
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    def _scan(self, data):
        """Apply blur, shared-channel motion, and contrast without RGB offsets."""
        img = data["image"]
        sigma = random.uniform(*SCAN_BLUR)
        img = self.scan_effect(img, sigma)
        length, angle = random.uniform(*SCAN_MOTION), random.uniform(0, 180)
        img = self.motion_effect(img, length, angle)
        contrast = random.uniform(*SCAN_CONTRAST)
        data["image"] = np.clip((img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
        self.last["скан"] = f"размытие σ {sigma:.2f} px"
        self.last["смаз"] = f"{length:.2f} px, угол {angle:.2f}°"
        self.last["контраст"] = f"{contrast:.2f}"
        return data

    def _noise(self, data):
        """Add shared-channel intensity noise after resampling and before JPEG."""
        img = data["image"]
        sigma = random.uniform(*SCAN_NOISE)
        shape = img.shape[:2] + (1,) if img.ndim == 3 else img.shape
        grain = np.random.normal(0, sigma, shape).astype(np.float32)
        data["image"] = np.clip(np.rint(img.astype(np.float32) + grain), 0, 255).astype(np.uint8)
        self.last["шум"] = f"гауссов, σ {sigma:.2f}"
        return data

    def _lowres(self, data):
        """Downsample and resize back, preserving dimensions while removing detail."""
        img = data["image"]
        h, w = img.shape[:2]
        if h <= MIN_H:
            return data
        nh = random.uniform(MIN_H, h)
        nw = max(1, round(w * nh / h))
        small = cv2.resize(img, (nw, round(nh)), interpolation=cv2.INTER_AREA)
        data["image"] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        self.last["высота"] = f"{h}->{round(nh)}"
        return data

    def _curl(self, data):
        """Simulate page curl with geometric parameters relative to crop height."""
        img = data["image"]
        h, w = img.shape[:2]
        amp, length = random.uniform(*CURL_AMP), random.uniform(*CURL_LEN)
        power, squeeze = random.uniform(*CURL_POWER), random.uniform(*CURL_SQUEEZE)
        side, up = random.choice(("right", "left", "both")), random.random() < 0.5
        L = max(1e-6, length * h)
        px = np.arange(w) + 0.5
        d = px - (w - L) if side == "right" else (L - px if side == "left" else np.abs(px - w / 2) - (w / 2 - L))
        u = np.clip(d / L, 0, 1)
        dy = amp * h * u ** power
        X = np.concatenate([[0], np.cumsum(1 - squeeze * u)])
        W2, H2 = max(2, int(np.ceil(X[-1]))), h + int(np.ceil(dy.max()))
        src_x = np.interp(np.arange(W2) + 0.5, X, np.arange(w + 1)) - 0.5
        dy_new = np.interp(src_x, np.arange(w), dy)
        Y = np.arange(H2, dtype=np.float32)[:, None]
        map_y = (Y - (H2 - h) + dy_new[None, :] if up else Y - dy_new[None, :]).astype(np.float32)
        map_x = np.broadcast_to(src_x[None, :], (H2, W2)).astype(np.float32)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light = gray > t
        paper = img[light].mean(axis=0) if light.any() else img.reshape(h * w, -1).max(axis=0)
        bg = tuple(float(c) for c in np.atleast_1d(paper))
        data["image"] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
        self.last["дуга"] = f"{side}, {'вверх' if up else 'вниз'}, изгиб {amp:.2f}, длина {length:.1f}, крутизна {power:.1f}, сжатие {squeeze:.2f}"
        return data

    def _jpeg(self, data):
        """Encode and decode JPEG at a random quality within JPEG_Q."""
        img = data["image"]
        q = random.randint(*JPEG_Q)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            data["image"] = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            self.last["jpeg"] = q
        return data

    def _erase(self, data):
        """Fade dark ink toward the paper color using a wear mask and strength in 0-1."""
        img = data["image"]
        h, w = img.shape[:2]
        L = img.astype(np.float32)
        if img.ndim == 2:
            L = L[:, :, None]
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        light = gray > t
        paper = L[light].mean(axis=0) if light.any() else L.reshape(-1, L.shape[2]).max(axis=0)
        i = random.randrange(len(self.pages))
        e = random.random()
        info = {}
        M = self._patch(self._mask(i), h, w, info)[:, :, None] / 255.0
        L, paper = to_lin(L), to_lin(paper)
        out = np.clip(to_srgb(L + (paper - L) * (M * e)), 0, 255).astype(np.uint8)
        data["image"] = out[:, :, 0] if img.ndim == 2 else out
        self.last.update({"стирание": self.names[i], "e": round(e, 2), "место_стирания": info})
        return data

    def _paper(self, data):
        """Multiply by a paper texture with a white point sampled from its median to 255."""
        img = data["image"]
        h, w = img.shape[:2]
        L = img.astype(np.float32)
        if img.ndim == 2:
            L = L[:, :, None]
        i = random.randrange(len(self.pages))
        info = {}
        P = self._patch(self._page(i), h, w, info)[:, :, None]
        wp = random.uniform(float(np.median(P)), 255.0)
        P = np.clip(P * (255.0 / wp), 0, 255)
        out = np.clip(to_srgb(to_lin(L) * to_lin(P)), 0, 255).astype(np.uint8)
        data["image"] = out[:, :, 0] if img.ndim == 2 else out
        self.last.update({"бумага": self.names[i], "белая_точка": round(wp, 1), "место_бумаги": info})
        return data


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input crop image or directory of crop images")
    ap.add_argument("--demo", type=int, default=10, help="Number of randomly selected crops")
    ap.add_argument("--out", required=True, help="Output directory for preview sheets")
    ap.add_argument("--variants", type=int, default=6, help="Augmented variants per crop")
    ap.add_argument("--font", help="Optional font file for preview captions")
    args = ap.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        paths = sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    else:
        ap.error("Input path does not exist")
    if not paths:
        ap.error("No input images found")
    if args.demo < 1 or args.variants < 1:
        ap.error("--demo and --variants must be positive")

    aug = TexAug()
    print("Textures:", len(aug.pages))
    font = ImageFont.truetype(args.font, 18) if args.font else ImageFont.load_default()
    os.makedirs(args.out, exist_ok=True)
    for k, path in enumerate(random.sample(paths, min(args.demo, len(paths)))):
        src = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if src is None:
            raise ValueError(f"Cannot decode input image: {path}")
        rows = [("Original", src)]
        for i in range(args.variants):
            out = aug({"image": src.copy()})["image"]
            rows.append((f"Variant {i + 1}" + ("  (unchanged)" if np.array_equal(out, src) else ""), out))
        S = max(1, round(64 / src.shape[0]))
        ims = [(t, Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).resize((im.shape[1] * S, im.shape[0] * S), Image.LANCZOS))
               for t, im in rows]
        W = max(max(i.width for _, i in ims), 600) + 20
        H = 40 + sum(i.height + 28 for _, i in ims)
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        d.text((10, 8), f"Crop {k + 1}", fill=(0, 0, 140), font=font)
        yy = 40
        for t, i in ims:
            d.text((10, yy), t, fill=(90, 90, 90), font=font)
            sheet.paste(i, (10, yy + 22))
            yy += i.height + 28
        sheet.save(os.path.join(args.out, f"{k:02d}.png"))
        print(k, path)

"""
Funções de corrupção sintética compartilhadas entre APDDv2Dataset (Exp 4)
e VideoFramesDataset (Exp 5a/5b).

Três tipos de ruído, intensidade 0-100:
    gaussian -> ruído gaussiano proporcional à intensidade
    blur     -> desfoque gaussiano proporcional à intensidade
    shapes   -> "pingos de tinta": manchas irregulares com cores amostradas
                da própria imagem, simulando corrupção por respingos
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

DEFAULT_NOISE_LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DEFAULT_NOISE_TYPES  = ["gaussian", "blur", "shapes"]

# Parâmetros do ruído "shapes" (pingos de tinta)
SHAPES_MAX_COUNT     = 8            # nº de manchas na intensidade máxima (100)
SHAPES_AREA_RANGE    = (0.02, 0.10) # fração da área da imagem ocupada por cada mancha
SHAPES_N_COLORS      = 5            # cores amostradas da própria imagem
SHAPES_IRREGULARITY  = 0.35         # variação do raio por vértice do blob


def add_gaussian_noise(image: Image.Image, intensity: int) -> Image.Image:
    sigma = 128 * intensity / 100
    arr = np.array(image.convert("RGB")).astype(np.int16)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_blur(image: Image.Image, intensity: int) -> Image.Image:
    radius = 20 * intensity / 100
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _sample_colors_from_image(img: Image.Image, n: int, rng: np.random.RandomState) -> list:
    """Amostra `n` cores de pixels aleatórios da própria imagem."""
    arr = np.array(img.convert("RGB"))
    h, w, _ = arr.shape
    ys = rng.randint(0, h, n)
    xs = rng.randint(0, w, n)
    return [tuple(int(c) for c in arr[y, x]) for y, x in zip(ys, xs)]


def _blob_polygon(cx: float, cy: float, radius: float, rng: np.random.RandomState,
                   n_points: int = 12, irregularity: float = SHAPES_IRREGULARITY) -> list:
    """Gera um polígono fechado irregular ao redor de (cx, cy) — formato de mancha/respingo."""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        r = radius * (1 + rng.uniform(-irregularity, irregularity))
        points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
    return points


def add_shapes(image: Image.Image, intensity: int,
                n_colors: int = SHAPES_N_COLORS,
                max_shapes: int = SHAPES_MAX_COUNT,
                area_range: tuple = SHAPES_AREA_RANGE) -> Image.Image:
    """
    Simula "pingos de tinta": desenha manchas irregulares sobre a imagem,
    usando cores amostradas da própria imagem. A quantidade de manchas
    cresce com a intensidade (0 -> nenhuma, 100 -> `max_shapes`), cada
    mancha ocupa entre `area_range` da área total, sem nunca cobrir a
    imagem por inteiro (cobertura total ~50% no nível máximo).
    """
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = np.random.RandomState(intensity)

    n_shapes = round(intensity / 100 * max_shapes)
    if intensity > 0:
        n_shapes = max(1, n_shapes)
    if n_shapes == 0:
        return img

    colors = _sample_colors_from_image(img, n_colors, rng)
    img_area = w * h

    for _ in range(n_shapes):
        area_frac = rng.uniform(*area_range)
        radius = float(np.sqrt(area_frac * img_area / np.pi))
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        color = colors[rng.randint(len(colors))]
        draw.polygon(_blob_polygon(cx, cy, radius, rng), fill=color)

    return img


NOISE_FNS = {
    "gaussian": add_gaussian_noise,
    "blur":     add_blur,
    "shapes":   add_shapes,
}

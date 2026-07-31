"""
Testa as funções de corrupção compartilhadas (datasets/noise.py),
usadas pelo APDDv2Dataset (Exp 4) e VideoFramesDataset (Exp 5a/5b).
"""
import numpy as np
from PIL import Image

from datasets.noise import add_gaussian_noise, add_blur, add_shapes, NOISE_FNS


def test_gaussian_level_zero_is_original(sample_image):
    result = add_gaussian_noise(sample_image, 0)
    assert np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))


def test_gaussian_changes_image(sample_image):
    result = add_gaussian_noise(sample_image, 80)
    assert not np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))
    assert result.size == sample_image.size


def test_blur_level_zero_is_original(sample_image):
    result = add_blur(sample_image, 0)
    assert np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))


def test_blur_changes_image(sample_image):
    result = add_blur(sample_image, 80)
    assert not np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))


def test_shapes_level_zero_is_original(sample_image):
    result = add_shapes(sample_image, 0)
    assert np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))


def test_shapes_changes_image(sample_image):
    result = add_shapes(sample_image, 100)
    assert not np.array_equal(np.array(sample_image.convert("RGB")), np.array(result))
    assert result.size == sample_image.size


def test_shapes_uses_colors_sampled_from_image():
    """Com uma imagem de cor única, toda mancha desenhada deve ter essa mesma cor."""
    solid_color = (10, 200, 30)
    img = Image.new("RGB", (64, 64), solid_color)
    result = add_shapes(img, 100)
    colors_present = set(result.getdata())
    assert colors_present == {solid_color}


def test_shapes_reproducible(sample_image):
    r1 = add_shapes(sample_image, 60)
    r2 = add_shapes(sample_image, 60)
    assert np.array_equal(np.array(r1), np.array(r2))


def test_shapes_coverage_grows_with_intensity(sample_image):
    """Mais intensidade -> mais pixels alterados em relação ao original."""
    orig = np.array(sample_image.convert("RGB"))
    low  = np.array(add_shapes(sample_image, 20))
    high = np.array(add_shapes(sample_image, 100))
    changed_low  = np.any(low != orig, axis=-1).sum()
    changed_high = np.any(high != orig, axis=-1).sum()
    assert changed_high >= changed_low


def test_shapes_never_fully_covers_image(sample_image):
    """No nível máximo (100), a cobertura não deve tomar a imagem inteira (~50% alvo)."""
    orig = np.array(sample_image.convert("RGB"))
    result = np.array(add_shapes(sample_image, 100))
    changed_frac = np.any(result != orig, axis=-1).mean()
    assert changed_frac < 0.85


def test_noise_fns_registry_has_all_types():
    assert set(NOISE_FNS.keys()) == {"gaussian", "blur", "shapes"}

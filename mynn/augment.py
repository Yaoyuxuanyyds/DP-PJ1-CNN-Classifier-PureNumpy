import cupy as cp
import cupyx.scipy.ndimage as cpx_ndimage
from tqdm import tqdm

def random_translate(images, max_shift=2):
    B = images.shape[0]
    shifts = cp.random.randint(-max_shift, max_shift, size=(B, 2))
    translated = cp.zeros_like(images)

    for i in tqdm(range(B), desc="Translating", leave=False):
        translated[i, 0] = cpx_ndimage.shift(
            images[i, 0],
            shift=tuple(shifts[i]),
            mode='constant',
            cval=0.0
        )
    translated = cp.clip(translated, 0, 1)
    return translated

def random_rotate(images, max_angle=5):
    B = images.shape[0]
    angles = cp.random.uniform(-max_angle, max_angle, size=B)
    rotated = cp.zeros_like(images)

    for i in tqdm(range(B), desc="Rotating", leave=False):
        rotated[i, 0] = cpx_ndimage.rotate(
            images[i, 0],
            angle=angles[i],
            reshape=False,
            mode='constant',
            cval=0.0
        )
    rotated = cp.clip(rotated, 0, 1)
    return rotated

def add_noise(images, noise_level=0.5):
    noise = cp.random.normal(0, noise_level, size=images.shape)
    return cp.clip(images + noise, 0, 1)

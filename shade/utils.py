import math
from PIL import Image


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    red_mask = Image.new("RGB", mask.size, color)
    mask = mask.convert("L")

    # Blend the red mask with the original image
    overlay = Image.composite(red_mask, image, mask)
    blended = Image.blend(image, overlay, alpha)

    return blended


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def resize_to_limit(image: Image.Image, megapixels: float) -> Image.Image:
    total_pixels = megapixels * 1_000_000
    if image.width * image.height <= total_pixels:
        return image

    aspect_ratio = image.width / image.height
    height = math.floor(math.sqrt(total_pixels / aspect_ratio))
    width = math.floor(height * aspect_ratio)

    resized_image = image.resize((width, height), Image.LANCZOS)
    return resized_image

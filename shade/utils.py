from PIL import Image


def overlay_mask_on_image(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    red_mask = Image.new("RGB", mask.size, color)
    mask = mask.convert("L")

    # Blend the red mask with the original image
    overlay = Image.composite(red_mask, image, mask)
    blended = Image.blend(image, overlay, alpha)

    return blended

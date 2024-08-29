import sys
from pathlib import Path
from PIL import Image

from shade.matter import Matter
from shade.segmenter import Segmenter
from shade.trimap_generator import TrimapGenerator
from shade.utils import overlay_mask_on_image

if len(sys.argv) < 2:
    print("No Image Defined: cli.py [image-file.png]\n\n")
    exit(0)

path = Path(sys.argv[1])

if path.is_file() == False:
    print("Invalid image")
    exit(0)

image = Image.open(path.absolute()).convert("RGB")


segmenter = Segmenter()
trimap_generator = TrimapGenerator(dilation_percentage=0.01, min_dilation=5)
matter = Matter()

print("Loading tools")
segmenter.init()
matter.init()

print("Done")


masks = segmenter.segment(image)

for mask in masks:
    print(f"Generating mask: {mask.label.name}")

    trimap = trimap_generator.generate(mask.image)
    mated_mask = matter.mate(mask.image, trimap)
    preview = overlay_mask_on_image(image, mated_mask)

    preview_path = Path(f"results/{mask.label.name}.png")
    mated_mask_path = Path(f"results/{mask.label.name}-mask.png")
    trimap_path = Path(f"results/{mask.label.name}-trimap.png")

    preview_path.parent.mkdir(parents=True, exist_ok=True)

    preview.save(preview_path.absolute())
    mated_mask.save(mated_mask_path.absolute())

    overlay_mask_on_image(image, trimap).save(trimap_path.absolute())

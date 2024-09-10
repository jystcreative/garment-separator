import sys
from pathlib import Path
from PIL import Image

from shade.matter import Matter
from shade.remove_bg import RemoveBG
from shade.segformer_segmenter import SegformerSegmenter
from shade.segment_task import SegmentTask
from shade.selfie_segmenter import SelfieSegmenter
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

task = SegmentTask()
task.init()
masks = task.run(image)

for mask in masks:

    mask_file = Path(f"results/{mask.label.name}-mask.png")
    mask.image.save(mask_file.absolute())

    preview_file = Path(f"results/{mask.label.name}.png")
    overlay_mask_on_image(image, mask.image).save(preview_file.absolute())

from typing import List
from shade.types import Label, Mask
from transformers import pipeline
from PIL import Image
from shade.utils import default_device


def map_segformer_label(label: str) -> Label:
    mapping = {
        'Background': Label.BACKGROUND,
        'Hair': Label.HAIR,
        'Face': Label.FACE_SKIN,
        'Upper-clothes': Label.UPPER_CLOTHES,
        'Pants': Label.PANTS,
        'Left-shoe': Label.LEFT_SHOE,
        'Right-shoe': Label.RIGHT_SHOE,
        'Left-leg': Label.LEFT_LEG,
        'Right-leg': Label.RIGHT_LEG,
        'Left-arm': Label.LEFT_ARM,
        'Right-arm': Label.RIGHT_ARM,
    }
    return mapping.get(label, Label.OTHER)


class SegformerSegmenter():
    def init(self):
        self.pipe = pipeline("image-segmentation",
                             model="mattmdjaga/segformer_b2_clothes",
                             device=default_device())

    def segment(self, image: Image.Image) -> List[Mask]:
        layers = self.pipe(image)

        return [
            Mask(
                label=map_segformer_label(layer['label']),
                image=layer['mask']
            )
            for layer in layers
        ]

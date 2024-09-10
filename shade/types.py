from enum import Enum
from dataclasses import dataclass
from PIL import Image


class Label(Enum):
    BACKGROUND = 'background'
    HAIR = 'hair'

    # ======= SKIN ========
    SKIN = 'skin'
    FACE_SKIN = 'skin.face'

    BODY_SKIN = 'skin.body'

    LEG = 'skin.body.leg'
    LEFT_LEG = 'skin.body.leg.left'
    RIGHT_LEG = 'skin.body.leg.left'

    ARM = 'skin.body.arm'
    LEFT_ARM = 'skin.body.arm.left'
    RIGHT_ARM = 'skin.body.arm.left'

    # ======= CLOTHES ========
    CLOTHES = 'clothes'
    UPPER_CLOTHES = 'clothes.upper'

    LOWER_CLOTHES = 'clothes.lower'
    SKIRT = 'clothes.lower.skirt'
    PANTS = 'clothes.lower.pants'

    SHOES = 'clothes.shoes'
    LEFT_SHOE = 'clothes.shoes.left'
    RIGHT_SHOE = 'clothes.shoes.right'

    OTHER = 'other'


@dataclass()
class Mask():
    label: Label
    image: Image.Image

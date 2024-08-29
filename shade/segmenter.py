from dataclasses import dataclass
from typing import List
import numpy as np
import mediapipe as mp

from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from urllib.request import urlretrieve
from pathlib import Path
from enum import Enum


class Label(Enum):
    BACKGROUND = 0
    HAIR = 1
    BODY_SKIN = 2
    FACE_SKIN = 3
    CLOTHES = 4
    OTHER = 5


@dataclass()
class Result():
    label: Label
    image: Image.Image


MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite?v=alkali.mediapipestudio_20240828_0657_RC00'
MODEL_PATH = './.models/model.tflite'


class Segmenter():
    def init(self):
        path = Path(MODEL_PATH)

        if path.is_file() == False:
            print('Downloading model...')
            path.parent.mkdir(parents=True, exist_ok=True)

            urlretrieve(MODEL_URL, path.absolute())
            print('Model Downloaded')

        base_options = python.BaseOptions(model_asset_path=path.absolute())
        self.options = vision.ImageSegmenterOptions(
            base_options=base_options, output_category_mask=True)

    def segment(self, image: Image.Image) -> List[Result]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.asarray(image))
        with vision.ImageSegmenter.create_from_options(self.options) as segmenter:
            result = segmenter.segment(mp_image)

            return [
                Result(
                    label=Label(i),
                    image=Image.fromarray(
                        (255 * mask.numpy_view()).astype(np.uint8))
                )
                for i, mask in enumerate(result.confidence_masks)
            ]

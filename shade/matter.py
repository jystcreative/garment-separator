import torch
import torchvision.transforms as T

from dataclasses import dataclass
from transformers import VitMatteImageProcessor, VitMatteForImageMatting
from PIL.ImageOps import fit, pad
from PIL import Image


@dataclass
class Matter():
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kernel_size: tuple[int, int] = (2048, 2048)
    model_name: str = 'hustvl/vitmatte-small-composition-1k'

    def init(self):
        self.processor = VitMatteImageProcessor.from_pretrained(
            self.model_name)

        self.model = VitMatteForImageMatting.from_pretrained(
            self.model_name).to(self.device)

    def mate(self, image: Image.Image, trimap_image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        trimap_image = trimap_image.convert("L")

        input_image = pad(image, self.kernel_size)
        input_trimap = pad(trimap_image, self.kernel_size)

        inputs = self.processor(
            images=input_image,
            trimaps=input_trimap,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            alphas = self.model(**inputs).alphas

        mask = T.ToPILImage()(torch.squeeze(alphas))

        # make sure sizes are same
        mask = fit(mask, image.size, method=Image.LANCZOS, bleed=0)
        return mask

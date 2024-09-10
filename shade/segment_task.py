import logging

from dataclasses import dataclass, field
from typing import List
from PIL import Image

from shade.matter import Matter
from shade.segformer_segmenter import SegformerSegmenter
from shade.trimap_generator import TrimapGenerator
from shade.types import Label, Mask
from shade.utils import overlay_mask_on_image

DEFAULT_LABELS = [Label.SUBJECT, Label.UPPER_CLOTHES]

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class SegmentTask():
    refine_with_subject = False
    matte_masks = True
    debug_trimap_to = 'results/trimap.png'

    labels: List[Label] = field(default_factory=lambda: DEFAULT_LABELS)

    segmenter: SegformerSegmenter = field(default_factory=SegformerSegmenter)
    matter: Matter = field(default_factory=Matter)

    def init(self):
        self.segmenter.init()
        self.matter.init()
        logger.info("Segment Task Initialized")

    def _clip(self, mask: Mask, master: Mask) -> Mask:
        clipped_mask = Image.composite(
            mask.image, Image.new("L", mask.image.size, 0), master.image)
        return Mask(label=mask.label, image=clipped_mask)

    def _mate_mask(self, image: Image.Image, mask: Mask) -> Mask:
        logger.info(f"Mating mask {mask.label}")

        trimap_generator = TrimapGenerator(
            dilation_percentage=0.02, min_dilation=5)

        trimap = trimap_generator.generate(mask.image)
        if self.debug_trimap_to:
            overlay_mask_on_image(image, trimap).save(self.debug_trimap_to)

        result = self.matter.mate(mask.image, trimap)

        return Mask(label=mask.label, image=result)

    def run(self, image: Image.Image) -> List[Mask]:
        logger.info("Running Segment Task")
        masks = self.segmenter.segment(image)
        masks = [mask for mask in masks if mask.label in self.labels]

        logger.info(f"Initial masks segmented {masks}")

        if self.matte_masks:
            masks = [self._mate_mask(image, mask) for mask in masks]

        subject = Mask(label=Label.SUBJECT, image=Image.open(
            "./removebg.png").convert("L"))

        if self.refine_with_subject:
            logger.info("Refining masks")
            masks = [self._clip(mask, subject) for mask in masks]

        return masks

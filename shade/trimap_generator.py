import cv2
import numpy as np

from dataclasses import dataclass
from PIL import Image
from typing import Optional


@dataclass
class TrimapGenerator:
    force_binary: bool = True
    binary_threshold: int = 127

    dilation_percentage: Optional[float] = None
    dilation: int = 100
    min_dilation: int = 2

    def generate(self, image: Image.Image) -> Image.Image:
        if self.dilation_percentage is not None:
            self.dilation = round(image.width * self.dilation_percentage)
            self.dilation = max(self.min_dilation, self.dilation)

        # Convert image to numpy array
        trimap = np.asarray(image.convert("L"))

        # Apply binary threshold if required
        if self.force_binary:
            _, trimap = cv2.threshold(
                trimap, self.binary_threshold, 255, cv2.THRESH_BINARY)

        # Detect edges
        edges = cv2.Canny(trimap, 100, 200)

        # Dilate the edges
        kernel = np.ones((self.dilation, self.dilation), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        trimap[dilated_edges > 0] = 128

        # Return the processed trimap as an image
        return Image.fromarray(trimap).convert("L")

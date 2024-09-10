import io
import os
import requests
from PIL import Image
from shade.types import Label, Mask
from shade.utils import resize_to_limit


class RemoveBG:
    API_ENDPOINT = 'https://api.remove.bg/v1.0/removebg'
    API_KEY = os.getenv('REMOVE_BG_API_KEY')

    def _image_to_blob(self, image: Image.Image) -> io.BytesIO:
        blob = io.BytesIO()
        image.save(blob, format='PNG')
        blob.seek(0)
        return blob

    def process(self, image: Image.Image) -> Mask:
        resized_image = resize_to_limit(image, 10)

        blob = self._image_to_blob(resized_image)
        form_data = {'size': 'auto'}
        files = {'image_file': ('image.png', blob, 'image/png')}

        response = requests.post(self.API_ENDPOINT, headers={
                                 'X-Api-Key': self.API_KEY}, data=form_data, files=files)

        if response.status_code != 200:
            raise Exception(f"Error with remove.bg API: {
                            response.status_code}, {response.text}")

        mask_image = Image.open(io.BytesIO(response.content))
        mask_resized = mask_image.resize(
            (image.width, image.height), Image.LANCZOS)

        alpha_channel = mask_resized.split()[-1]
        return Mask(label=Label.SUBJECT, image=alpha_channel)

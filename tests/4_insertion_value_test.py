from incx.tracking.incx import IncX
from incx.models.model_enum import ModelEnum
from incx.models.model_factory import ModelFactory
from incx.explainers.d_rise import DRise
import numpy as np
from PIL import Image
import cv2
from incx.metrics.saliency_maps.insertion import compute_insertion


def test_insertion_value():
    # Given
    image_locations = [f"datasets/LASOT/1/{str(i).zfill(8)}.jpg" for i in range(1, 10)]
    images = [
        resize_image(image_location, (640, 480)) for image_location in image_locations
    ]
    model = ModelFactory().get_model(ModelEnum.YOLO)
    explainer = DRise(model, 500)
    incRex = IncX(model, explainer)

    # When
    average_insertion = 0
    for i, image in enumerate(images):
        results, _ = incRex.explain_frame(image)
        result = results[0]
        insertion = compute_insertion(
            model, result.saliency_map, image, 2, result.bounding_box, divisions=100
        )
        if i != 0:
            average_insertion += insertion
    average_insertion = average_insertion / (len(images) - 1)

    # Then
    assert average_insertion > 0.5


def resize_image(image_path, target_size):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    cv2_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

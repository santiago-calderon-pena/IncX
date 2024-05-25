
from vision_explanation_methods.explanations import common as od_common
import numpy as np
import torch

import numpy as np

class PytorchYoloV8Wrapper(od_common.GeneralObjectDetectionModelWrapper):
    """Wraps a PytorchFasterRCNN model with a predict API function for object detection.

    To be compatible with the drise explainability method, all models must be wrapped to have
    the same output and input class.
    This wrapper is customized for the FasterRCNN model from Pytorch, and can
    also be used with the RetinaNet or any other models with the same output class.
    """
    
    def __init__(self, model):
        self._model = model
        self._number_of_classes = 80

    def predict(self, x: torch.Tensor):
        """Creates a list of detection records from the image predictions.
        """
        raw_detections = []
        for x_el in x:
            input  = (np.ascontiguousarray(np.transpose(x_el.numpy(), (1, 2, 0))) * 255).astype('uint8')
            raw_detection = self._model.predict(input, verbose = False)
            raw_detections.append(raw_detection)
        
        detections = [] 
        for raw_detection in raw_detections:
            #raw_detection = apply_nms(raw_detection,0.005)
            
            # Note that FasterRCNN doesn't return a score for each class, only the predicted class
            # DRISE requires a score for each class. We approximate the score for each class
            # by dividing the (1.0 - class score) evenly among the other classes.
            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detection[0].boxes.xyxy,
                    class_scores=od_common.expand_class_scores(raw_detection[0].boxes.conf,
                                                                  raw_detection[0].boxes.cls,
                                                                  self._number_of_classes),
                    objectness_scores=torch.tensor([1.0]*raw_detection[0].boxes.conf.shape[0]),
                )
            )
        
        return detections

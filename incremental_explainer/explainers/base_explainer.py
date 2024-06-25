from abc import ABC, abstractmethod
from vision_explanation_methods.explanations import common as od_common

class BaseExplainer(ABC):
    
    @abstractmethod
    def create_saliency_map(self, results, image_location, model: od_common.GeneralObjectDetectionModelWrapper):
        pass
from vision_explanation_methods import DRISE_runner as dr
from helpers.pytorch_yolov8_wrapper import PytorchYoloV8Wrapper
from ultralytics import YOLO

def compute_saliency_maps(results, image_location, model, nummasks=50):
    # print(image_location)
    # print(results)
    number = 0
    while (len(results[0].boxes.cls) != number): # Check assumption that it sometimes cannot find the same number of objects
        results_drise = dr.get_drise_saliency_map(nummasks=nummasks, imagelocation=image_location, model= PytorchYoloV8Wrapper(model), savename="anything", numclasses=80, max_figures=2, maskres=(16,16))
        number = len(results_drise)
    # print("Number of objects found in expl: ", number)
    # print("Number of objects found in bounding box: ", len(results[0].boxes.cls))
    return results_drise
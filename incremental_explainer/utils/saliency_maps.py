from vision_explanation_methods import DRISE_runner as dr

from vision_explanation_methods.explanations import common as od_common

def compute_saliency_maps(results, image_location, model: od_common.GeneralObjectDetectionModelWrapper, nummasks=50):

    number = 0
    while (len(results[0].bounding_boxes) != number):
        results_drise = dr.get_drise_saliency_map(nummasks=nummasks, imagelocation=image_location, model= model, savename="anything", numclasses=80, max_figures=2, maskres=(16,16))
        number = len(results_drise)

    return results_drise
from math import ceil, floor
import numpy as np


def compute_energy_based_pointing_game(saliency_map, bounding_box):

    x_0, y_0, x_1, y_1 = bounding_box
    mask = np.zeros_like(saliency_map)
    mask[floor(y_0):ceil(y_1), floor(x_0):ceil(x_1)] = 1
    saliency_map_masked = saliency_map * mask
    return sum([sum(row) for row in saliency_map_masked]) / sum([sum(row) for row in saliency_map])
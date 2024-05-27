import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from numpy import trapz
    
def compute_insertion_array(model, saliency_map, image, class_index, divisions=100):
        import matplotlib as mpl
        mpl.rcParams["savefig.pad_inches"] = 0
        masks = np.empty([saliency_map.shape[0], saliency_map.shape[1], 3])
        conf_insertion_list = []
        divisions_list_in = []
        minimum = np.min(saliency_map) - np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.2
        )
        maximum = np.max(saliency_map) + np.abs(
            (np.min(saliency_map) - np.max(saliency_map)) * 0.1
        )
        for sub_index in range(0, divisions):
            masks[:, :, :] = False
            threshold = maximum + (sub_index / divisions) * (minimum - maximum)
            pixels = np.where(saliency_map >= threshold)
            masks[pixels[0], pixels[1], :] = True
            divisions_list_in.append(
                len(pixels[0]) / (saliency_map.shape[0] * saliency_map.shape[1])
            )
            min_expl = np.where(masks, image, 0)
            detection = model(min_expl, classes=class_index, verbose=False)
            conf = (
                detection[0].boxes.conf[0]
                if (len(detection[0].boxes.conf) > 0)
                else detection[0].boxes.conf
            )
            print(sub_index)
            # plt.axis('off')
            # plt.imshow(min_expl)
            # plt.show()
            conf_insertion_list.append(conf)
        conf_insertion_list = [
            np.sum(conf.numpy()) if conf.numpy().size > 0 else 0
            for conf in conf_insertion_list
        ]
    
        return conf_insertion_list, divisions_list_in, trapz(conf_insertion_list, divisions_list_in)
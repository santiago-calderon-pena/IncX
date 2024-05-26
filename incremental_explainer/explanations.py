import numpy as np

def compute_explanation(img, class_id, exp, first_level, model):
    if(first_level == -1):
        x = np.flip(np.arange(0, exp.shape[1]))
        y = np.arange(0, exp.shape[0])
        X, Y = np.meshgrid(x, y)

        levels = np.flip(np.unique(exp))
        masks = np.empty([exp.shape[0], exp.shape[1], 3])
        min_expl = []
        for level in levels:
            pixels = np.where(exp == level)
            masks[pixels[0], pixels[1], :] = True
            min_expl = np.where(masks, img, 0)
            pre = model.predict([min_expl])
            if (class_id not in pre[0].boxes.cls.numpy()):
                continue
            break
        first_level = level
    else:
        pixels = np.where(exp >= first_level)
        masks = np.empty([exp.shape[0], exp.shape[1], 3])
        masks[pixels[0], pixels[1], :] = True
        min_expl = np.where(masks, img, 0)
    return min_expl, first_level
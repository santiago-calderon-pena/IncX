from tqdm import tqdm
import pickle
from incx.models.model_enum import ModelEnum
from incx.metrics.saliency_maps.deletion import compute_deletion
from incx.metrics.saliency_maps.insertion import compute_insertion
from incx.metrics.saliency_maps.epg import compute_energy_based_pointing_game
from incx.metrics.saliency_maps.exp_proportion import compute_explanation_proportion
from collections import defaultdict
from IPython.display import clear_output
import random
import joblib
from filelock import FileLock
import numpy as np
from incx.models.model_factory import ModelFactory
from PIL import Image
import os
from dotenv import load_dotenv
def default_float_dict():
    return defaultdict(float)

def main():
    load_dotenv()
    INCX_RESULTS_FOLDER_PATH = os.environ.get("INCX_RESULTS_FOLDER_PATH")
    D_RISE_RESULTS_FOLDER_PATH = os.environ.get("D_RISE_RESULTS_FOLDER_PATH")

    blob_name_file_lock = "blob_names_metrics.lock"
    lock_blobs_name_comparison = FileLock(blob_name_file_lock, timeout=100)

    comparison_file_lock = "metrics_comparison.lock"
    lock_comparison = FileLock(comparison_file_lock, timeout=100)
    
    with lock_blobs_name_comparison:
        blob_names_incx = joblib.load("blob_names_metrics.pkl")

    model_factory = ModelFactory()

    while blob_names_incx:
        with lock_blobs_name_comparison:
            blob_name = blob_names_incx.pop()
            joblib.dump(blob_names_incx, "blob_names_metrics.pkl")

        dict_incx = joblib.load(INCX_RESULTS_FOLDER_PATH + '/' + blob_name)
        dict_drise = joblib.load(D_RISE_RESULTS_FOLDER_PATH + '/' + blob_name)

        dataset_name = blob_name.split("/")[0]
        model_name = blob_name.split("/")[2]
        video_number = blob_name.split("/")[3]
        image_name = blob_name.split("/")[4].split(".")[0]
        current_index = dict_incx["detection"]["current_index"]
        class_index = dict_incx["detection"]["class_index"]
        saliency_map_incx = dict_incx["maps"]["saliency_map"]
        saliency_map_drise = dict_drise["maps"]["saliency_map"]
        model_enum_el = ModelEnum[model_name]
        model = model_factory.get_model(model_enum_el)
        image_path = f"datasets/{dataset_name}/{video_number}/{image_name}"
        print(image_path)
        image_path = image_path + ".png" if os.path.exists(image_path + ".png") else image_path + ".jpg"
        image = np.array(Image.open(image_path).convert("RGB"))
        bounding_box = dict_drise["detection"]["bounding_box"]

        if class_index > 80:
            continue

        deletion_drise = compute_deletion(
            model=model,
            saliency_map=saliency_map_drise,
            image=image,
            class_index=class_index,
            bounding_box=bounding_box,
            object_index=current_index,
            divisions=100,
        )
        deletion_incx = compute_deletion(
            model=model,
            saliency_map=saliency_map_incx,
            image=image,
            class_index=class_index,
            bounding_box=bounding_box,
            object_index=current_index,
            divisions=100,
        )
        insertion_drise = compute_insertion(
            model=model,
            saliency_map=saliency_map_drise,
            image=image,
            class_index=class_index,
            bounding_box=bounding_box,
            object_index=current_index,
            divisions=100,
        )
        insertion_incx = compute_insertion(
            model=model,
            saliency_map=saliency_map_incx,
            image=image,
            class_index=class_index,
            bounding_box=bounding_box,
            object_index=current_index,
            divisions=100,
        )
        epg_drise = compute_energy_based_pointing_game(
            saliency_map=saliency_map_drise, bounding_box=bounding_box
        )
        epg_incx = compute_energy_based_pointing_game(
            saliency_map=saliency_map_incx, bounding_box=bounding_box
        )
        time_drise = dict_drise["metrics"]["explanation_time"]
        time_incx = dict_incx["metrics"]["explanation_time"]
        mask_incx = dict_incx["maps"]["mask"]
        mask_drise = dict_drise["maps"]["mask"]
        exp_proportion_drise = compute_explanation_proportion(mask_drise)
        exp_proportion_incx = compute_explanation_proportion(mask_incx)

        frame_name = blob_name.split("/")[-1].split(".")[0]
        video_number = blob_name.split("/")[-2]
        with lock_comparison:
            metrics_results = joblib.load("metrics_results.pkl")
            metrics_results["D-RISE"][dataset_name][model_name]["Insertion"][video_number][
                frame_name
            ] = insertion_drise
            metrics_results["D-RISE"][dataset_name][model_name]["Deletion"][video_number][
                frame_name
            ] = deletion_drise
            metrics_results["D-RISE"][dataset_name][model_name]["EPG"][video_number][frame_name] = (
                epg_drise
            )
            metrics_results["D-RISE"][dataset_name][model_name]["Explanation Proportion"][video_number][
                frame_name
            ] = exp_proportion_drise
            metrics_results["D-RISE"][dataset_name][model_name]["Time"][video_number][frame_name] = (
                time_drise
            )
            metrics_results["Incx"][dataset_name][model_name]["Insertion"][video_number][frame_name] = (
                insertion_incx
            )
            metrics_results["Incx"][dataset_name][model_name]["Deletion"][video_number][frame_name] = (
                deletion_incx
            )
            metrics_results["Incx"][dataset_name][model_name]["EPG"][video_number][frame_name] = (
                epg_incx
            )
            metrics_results["Incx"][dataset_name][model_name]["Explanation Proportion"][video_number][
                frame_name
            ] = exp_proportion_incx
            metrics_results["Incx"][dataset_name][model_name]["Time"][video_number][frame_name] = (
                time_incx
            )
            joblib.dump(metrics_results, "metrics_results.pkl")

        clear_output(wait=True)
        print(f"Deletion DRiSE: {deletion_drise}")
        print(f"Deletion IncX: {deletion_incx}")
        print(f"Insertion DRiSE: {insertion_drise}")
        print(f"Insertion IncX: {insertion_incx}")
        print(f"EPG DRiSE: {epg_drise}")
        print(f"EPG IncX: {epg_incx}")
        print(f"Exp Proportion DRiSE: {exp_proportion_drise}")
        print(f"Exp Proportion IncX: {exp_proportion_incx}")

        with lock_blobs_name_comparison:
            blob_names_incx = joblib.load("blob_names_metrics.pkl")
        random.shuffle(blob_names_incx)
        print(f"Missing: {len(blob_names_incx)}")

if __name__ == "__main__":
    main()
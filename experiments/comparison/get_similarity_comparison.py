import os
import random
import joblib
import pickle
from dotenv import load_dotenv
from filelock import FileLock
from azure.storage.blob import BlobServiceClient
from incx.metrics.comparison.explanations.dice_coefficient import (
    compute_dice_coefficient,
)
from incx.metrics.comparison.explanations.jaccard_index import (
    compute_jaccard_index,
)
from incx.metrics.comparison.saliency_maps.pearson_coefficient import (
    compute_pearson_coefficient,
)
from incx.metrics.comparison.saliency_maps.structural_similarity_index import (
    compute_structural_similarity_index,
)
from IPython.display import clear_output
from collections import defaultdict
def default_float_dict():
    return defaultdict(float)

def main():
    # Load environment variables
    load_dotenv()

    INCX_RESULTS_FOLDER_PATH = os.environ.get("INCX_RESULTS_FOLDER_PATH")
    D_RISE_RESULTS_FOLDER_PATH = os.environ.get("D_RISE_RESULTS_FOLDER_PATH")
    # Setup file locks
    blob_name_file_lock = "blob_names_comparison.lock"
    lock_blobs_name_comparison = FileLock(blob_name_file_lock, timeout=100)

    comparison_file_lock = "comparison.lock"
    lock_comparison = FileLock(comparison_file_lock, timeout=100)
    # Load blob names
    with lock_blobs_name_comparison:
        blob_names_incx = joblib.load("blob_names.pkl")

    random.shuffle(blob_names_incx)
    # Process each blob
    num_blobs = 0
    while blob_names_incx:
        with lock_blobs_name_comparison:
            blob_name = blob_names_incx.pop()
            joblib.dump(blob_names_incx, "blob_names.pkl")
        print(f"Processing blob {blob_name}")

        clear_output(wait=True)

        # Download and load blob data for INCX
        dict_incx = joblib.load(INCX_RESULTS_FOLDER_PATH +'/'+blob_name)
        model_name = blob_name.split("/")[2]
        dataset_name = blob_name.split("/")[0]
        print(f"Model: {model_name}, Dataset: {dataset_name}")
        # Download and load blob data for DRISE
        dict_drise = joblib.load(D_RISE_RESULTS_FOLDER_PATH +'/'+blob_name)

        # Extract maps for comparison
        saliency_map_incx = dict_incx["maps"]["saliency_map"]
        saliency_map_drise = dict_drise["maps"]["saliency_map"]

        # Compute metrics
        pearson_coeff = compute_pearson_coefficient(
            saliency_map_incx, saliency_map_drise
        )
        structural_similarity_index = compute_structural_similarity_index(
            saliency_map_incx, saliency_map_drise
        )

        mask_incx = dict_incx["maps"]["mask"]
        mask_drise = dict_drise["maps"]["mask"]

        dice_coefficient = compute_dice_coefficient(mask_incx, mask_drise)
        jaccard_index = compute_jaccard_index(mask_incx, mask_drise)

        num_blobs += 1

        # Update results
        frame_name = blob_name.split("/")[-1].split(".")[0]
        video_number = blob_name.split("/")[-2]
        print(f"Frame: {frame_name}, Video: {video_number}")
        with lock_comparison:
            comparison_results = joblib.load("comparison_results.pkl")
            comparison_results[dataset_name][model_name]["Pearson"][video_number][frame_name] = (
                pearson_coeff
            )
            comparison_results[dataset_name][model_name]["Structural"][video_number][frame_name] = (
                structural_similarity_index
            )
            comparison_results[dataset_name][model_name]["Dice"][video_number][frame_name] = (
                dice_coefficient
            )
            comparison_results[dataset_name][model_name]["Jaccard"][video_number][frame_name] = (
                jaccard_index
            )
            joblib.dump(comparison_results, "comparison_results.pkl")

        print(f"Finished blob {blob_name}")
        with lock_blobs_name_comparison:
            blob_names_incx = joblib.load("blob_names.pkl")
        random.shuffle(blob_names_incx)
        print(f"Missing: {len(blob_names_incx)}")


if __name__ == "__main__":
    main()

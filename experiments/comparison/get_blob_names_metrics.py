import os
import random
import joblib
from dotenv import load_dotenv
from incx.models.model_enum import ModelEnum
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from datasets.dataset_enum import DatasetEnum
import numpy as np
from collections import defaultdict

def find_files(directory="."):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def main():
    load_dotenv()
    INCX_RESULTS_FOLDER_PATH = os.environ.get("INCX_RESULTS_FOLDER_PATH")
    
    blob_names = find_files(INCX_RESULTS_FOLDER_PATH)
    
    blob_names = [blob_name.replace("\\", "/") for blob_name in blob_names]
    blob_names = [
        '/'.join(name.split('/')[-5:]) 
        for name in blob_names
    ]
    random.shuffle(blob_names)

    joblib.dump(blob_names, "blob_names_metrics.pkl")

    results = {
        "D-RISE": {
            dataset.name:{
                model.name: {
                    "Insertion": defaultdict(lambda: defaultdict(float)),
                    "Deletion": defaultdict(lambda: defaultdict(float)),
                    "EPG": defaultdict(lambda: defaultdict(float)),
                    "Explanation Proportion": defaultdict(lambda: defaultdict(float)),
                    "Time": defaultdict(lambda: defaultdict(float)),
                } for model in ModelEnum
            } for dataset in DatasetEnum
        },
        "Incx": {
            dataset.name:{
                model.name: {
                    "Insertion": defaultdict(lambda: defaultdict(float)),
                    "Deletion": defaultdict(lambda: defaultdict(float)),
                    "EPG": defaultdict(lambda: defaultdict(float)),
                    "Explanation Proportion": defaultdict(lambda: defaultdict(float)),
                    "Time": defaultdict(lambda: defaultdict(float)),
                } for model in ModelEnum
            } for dataset in DatasetEnum
        },
    }
    joblib.dump(results, "metrics_results.pkl")

    print(
        "Blob names have been saved to 'blob_names_metrics.pkl and comparison results array has been initialized 'metrics_results.pkl'."
    )


if __name__ == "__main__":
    main()

import os
import joblib
from incx.models.model_enum import ModelEnum
from incx.explainers.explainer_enum import ExplainerEnum
from datasets.dataset_enum import DatasetEnum

def main():
    directory = "./datasets/"
    combinations_array = []
    for dataset in DatasetEnum:
        for explainer_name in ExplainerEnum:
            for model_name in ModelEnum:
                for dir in os.listdir(directory + dataset.name + "/"):
                    combinations_array.append((dataset, model_name, explainer_name, dir))

    joblib.dump(combinations_array, "jobs.pkl")


if __name__ == "__main__":
    main()

from dataclasses import asdict
from enum import StrEnum
import lightning
from bronze_age.config import BronzeConfig, LayerTypeBronze
from bronze_age.datasets import DatasetEnum
from train_bronze import train, get_config_for_dataset
import pandas as pd
import os

ALL_DATASETS = [
    DatasetEnum.INFECTION,
    DatasetEnum.SATURATION,
    DatasetEnum.BA_SHAPES,
    DatasetEnum.TREE_CYCLE,
    DatasetEnum.TREE_GRID,
    DatasetEnum.BA_2MOTIFS,
    # Algorithmic dataset
    DatasetEnum.DISTANCE,
    DatasetEnum.PATH_FINDING,
    DatasetEnum.PREFIX_SUM,
    DatasetEnum.ROOT_VALUE,
    DatasetEnum.GAME_OF_LIFE,
    DatasetEnum.HEXAGONAL_GAME_OF_LIFE,
    # Real-world datasets
    DatasetEnum.MUTAG,
    DatasetEnum.MUTAGENICITY,
    DatasetEnum.BBBP,
    DatasetEnum.PROTEINS,
    DatasetEnum.IMDB_BINARY,
    DatasetEnum.REDDIT_BINARY,
    DatasetEnum.COLLAB,
]

SYNTHETIC_DATASETS = [
    DatasetEnum.INFECTION,
    DatasetEnum.SATURATION,
    DatasetEnum.BA_SHAPES,
    DatasetEnum.TREE_CYCLE,
    DatasetEnum.TREE_GRID,
    DatasetEnum.BA_2MOTIFS,
]

NAR_DATASETS = [
    DatasetEnum.DISTANCE,
    DatasetEnum.PATH_FINDING,
    DatasetEnum.PREFIX_SUM,
    DatasetEnum.ROOT_VALUE,
    DatasetEnum.GAME_OF_LIFE,
    DatasetEnum.HEXAGONAL_GAME_OF_LIFE,
]

EXPERIMENT_DIR = "experiments/"


def save_data_to_csv(dicts, filename):
    file_path = EXPERIMENT_DIR + filename
    # Create the directory if it doesn't exist
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    df = pd.DataFrame(dicts)
    df.to_csv(file_path, index=False)


def run_experiment(experiment_title, datasets, **config_args):
    results = []
    start_time = pd.Timestamp.now().strftime("%d_%m_%y_%H_%M")

    for dataset in datasets:
        try:
            print(f"Experiment: {experiment_title}, Dataset: {dataset}")
            print("Settings: ")
            print(config_args)
            config: BronzeConfig = get_config_for_dataset(dataset, **config_args)

            config_dict = asdict(config)
            for key, value in config_dict.items():
                if isinstance(value, StrEnum):
                    config_dict[key] = value.value

            lightning.seed_everything(0, workers=True)
            (
                test_accuracies,
                test_accuracies_dt,
                test_accuracies_dt_pruned,
                test_accuracies_cm,
            ) = train(config, base_experiment_title=experiment_title)
            for i in range(len(test_accuracies)):
                test_accuracy = test_accuracies[i]
                if config.train_decision_tree:
                    test_accuracy_dt = test_accuracies_dt[i]
                    test_accuracy_dt_pruned = test_accuracies_dt_pruned[i]
                else:
                    test_accuracy_dt = None
                    test_accuracy_dt_pruned = None
                if config.train_concept_model:
                    test_accuracy_cm = test_accuracies_cm[i]
                else:
                    test_accuracy_cm = None

                result_entry = {
                    "experiment": experiment_title,
                    "start_time": start_time,
                    "dataset": dataset.value,
                    "fold": i,
                    "test_accuracy": test_accuracy,
                    "test_accuracy_dt": test_accuracy_dt,
                    "test_accuracy_dt_pruned": test_accuracy_dt_pruned,
                    "test_accuracy_cm": test_accuracy_cm,
                    **config_dict,
                }

                results.append(result_entry)
            save_data_to_csv(
                results, f"results_{experiment_title.replace(' ', '_')}_temp.csv"
            )

        except Exception as e:
            print(
                f"Experiment: {experiment_title}, Dataset: {dataset} failed with error: {e}"
            )
    save_data_to_csv(
        results, f"results_{experiment_title.replace(' ', '_')}_{start_time}.csv"
    )


if __name__ == "__main__":
    # Example usage
    experiment_title = "Ablation study - Bronze Age"
    run_experiment(
        experiment_title,
        [DatasetEnum.SIMPLE_SATURATION],
        layer_type=LayerTypeBronze.DEEP_CONCEPT_REASONER,
    )

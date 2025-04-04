import os
from dataclasses import asdict
from enum import StrEnum

import lightning
import pandas as pd

from bronze_age.config import (
    AggregationMode,
    BronzeConfig,
    LayerTypeBronze,
    LossMode,
    NonLinearity,
)
from bronze_age.datasets import DatasetEnum
from train_bronze import get_config_for_dataset, train

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
    DatasetEnum.SIMPLE_SATURATION,
    DatasetEnum.SATURATION,
    DatasetEnum.INFECTION,
    DatasetEnum.BA_SHAPES,
    DatasetEnum.TREE_CYCLE,
    DatasetEnum.TREE_GRID,
    DatasetEnum.BA_2MOTIFS,
]

REAL_WORLD_DATASETS = [
    DatasetEnum.MUTAG,
    DatasetEnum.MUTAGENICITY,
    DatasetEnum.BBBP,
    DatasetEnum.PROTEINS,
    DatasetEnum.IMDB_BINARY,
    DatasetEnum.REDDIT_BINARY,
    DatasetEnum.COLLAB,
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
            import traceback

            traceback.print_exc()
            continue
    save_data_to_csv(
        results, f"results_{experiment_title.replace(' ', '_')}_{start_time}.csv"
    )


def run_stoneage():
    experiment_title = "Final Results - Baseline - Stone Age"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 1000,
        "layer_type": LayerTypeBronze.MLP,
        "nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "evaluation_nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "concept_embedding_size": 128,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.0,
        "early_stopping": True,
        "loss_mode": LossMode.CROSS_ENTROPY,
        "train_decision_tree": True,
        "aggregation_mode": AggregationMode.STONE_AGE,
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(
        experiment_title, SYNTHETIC_DATASETS + REAL_WORLD_DATASETS, **specific_config
    )


def run_bronzeage_dcr():
    experiment_title = "Final Results - Bronze Age DCR"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 10,
        "layer_type": LayerTypeBronze.DEEP_CONCEPT_REASONER,
        "nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": 128,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.2,
        "early_stopping": True,
        "loss_mode": LossMode.BINARY_CROSS_ENTROPY,
        "train_decision_tree": False,
        "aggregation_mode": AggregationMode.BRONZE_AGE_COMPARISON,
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(
        experiment_title, SYNTHETIC_DATASETS + REAL_WORLD_DATASETS, **specific_config
    )


def run_bronzeage_cmr():
    experiment_title = "Final Results - Bronze Age CMR"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 10,
        "layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": 128,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.2,
        "early_stopping": True,
        "loss_mode": LossMode.BINARY_CROSS_ENTROPY,
        "train_decision_tree": False,
        "aggregation_mode": AggregationMode.BRONZE_AGE_COMPARISON,
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(
        experiment_title, SYNTHETIC_DATASETS + REAL_WORLD_DATASETS, **specific_config
    )


def run_ablation_experiment(
    experiment_title: str,
    layer_type: LayerTypeBronze,
    entropy_loss_scaling: float,
    aggregation_mode: AggregationMode,
    training_nonlinearity: NonLinearity | None = None,
    embedding_size: int = 32,
):
    specific_config = {
        "bounding_parameter": 10,
        "layer_type": layer_type,
        "nonlinearity": training_nonlinearity,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": embedding_size,
        "entropy_loss_scaling": entropy_loss_scaling,
        "train_decision_tree": False,
        "aggregation_mode": aggregation_mode,
        "train_concept_model": False,
    }
    run_experiment(experiment_title, SYNTHETIC_DATASETS, **specific_config)


def run_ablation_study(layer_type: LayerTypeBronze):
    standard_entropy_loss = 0.2
    layer_title = (
        "DCR" if layer_type == LayerTypeBronze.DEEP_CONCEPT_REASONER else "CMR"
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - Entropy Loss, Bronze Age",
        layer_type,
        entropy_loss_scaling=standard_entropy_loss,
        aggregation_mode=AggregationMode.BRONZE_AGE,
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - No Entropy Loss, Bronze Age",
        layer_type,
        entropy_loss_scaling=0.0,
        aggregation_mode=AggregationMode.BRONZE_AGE,
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - Entropy Loss, Rounded",
        layer_type,
        entropy_loss_scaling=standard_entropy_loss,
        aggregation_mode=AggregationMode.BRONZE_AGE_ROUNDED,
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - No Entropy Loss, Rounded",
        layer_type,
        entropy_loss_scaling=0.0,
        aggregation_mode=AggregationMode.BRONZE_AGE_ROUNDED,
    )

    run_ablation_experiment(
        f"Ablation Study - {layer_title} - Entropy Loss, Comparison",
        layer_type,
        entropy_loss_scaling=standard_entropy_loss,
        aggregation_mode=AggregationMode.BRONZE_AGE_COMPARISON,
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - No Entropy Loss, Comparison",
        layer_type,
        entropy_loss_scaling=0.0,
        aggregation_mode=AggregationMode.BRONZE_AGE_COMPARISON,
    )

    run_ablation_experiment(
        f"Ablation Study - {layer_title} - Entropy Loss, Bronze Age, Diff Argmax",
        layer_type,
        entropy_loss_scaling=standard_entropy_loss,
        aggregation_mode=AggregationMode.BRONZE_AGE,
        training_nonlinearity=NonLinearity.DIFFERENTIABLE_ARGMAX,
    )
    run_ablation_experiment(
        f"Ablation Study - {layer_title} - No Entropy Loss, Bronze Age, Diff Argmax",
        layer_type,
        entropy_loss_scaling=0.0,
        aggregation_mode=AggregationMode.BRONZE_AGE,
        training_nonlinearity=NonLinearity.DIFFERENTIABLE_ARGMAX,
    )
    if layer_type == LayerTypeBronze.DEEP_CONCEPT_REASONER:
        run_ablation_experiment(
            f"Ablation Study - {layer_title} - Entropy Loss, Bronze Age, 32 emb size",
            layer_type,
            entropy_loss_scaling=standard_entropy_loss,
            aggregation_mode=AggregationMode.BRONZE_AGE,
            training_nonlinearity=NonLinearity.DIFFERENTIABLE_ARGMAX,
            embedding_size=32,
        )


def run_simple_saturation_stone_age():
    experiment_title = "Simple Saturation - Stone Age"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 1000,
        "layer_type": LayerTypeBronze.MLP,
        "nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "evaluation_nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "concept_embedding_size": 128,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.0,
        "early_stopping": True,
        "loss_mode": LossMode.CROSS_ENTROPY,
        "train_decision_tree": True,
        "aggregation_mode": AggregationMode.STONE_AGE,
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(experiment_title, [DatasetEnum.SIMPLE_SATURATION], **specific_config)


def run_simple_saturation_bronze_age(layer_type: LayerTypeBronze):
    experiment_title = f"Simple Saturation - {layer_type}"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 10,
        "layer_type": layer_type,
        "nonlinearity": None,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": (
            32 if layer_type == LayerTypeBronze.DEEP_CONCEPT_REASONER else 128
        ),
        "concept_temperature": 0.5,
        "entropy_loss_scaling": 0.2,
        "early_stopping": False,
        "loss_mode": LossMode.BINARY_CROSS_ENTROPY,
        "train_decision_tree": False,
        "aggregation_mode": AggregationMode.BRONZE_AGE,
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 2,
    }
    run_experiment(experiment_title, [DatasetEnum.SIMPLE_SATURATION], **specific_config)


run_simple_saturation_bronze_age_dcr = lambda: run_simple_saturation_bronze_age(
    LayerTypeBronze.DEEP_CONCEPT_REASONER
)
run_simple_saturation_bronze_age_cmr = lambda: run_simple_saturation_bronze_age(
    LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER
)


def run_student_teacher_training_bronze():
    experiment_title = "Student Teacher Training - Bronze Age CMR"
    specific_config = {
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 10,
        "layer_type": LayerTypeBronze.MLP,
        "nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": 256,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.2,
        "early_stopping": True,
        "loss_mode": LossMode.CROSS_ENTROPY,
        "train_decision_tree": True,
        "aggregation_mode": AggregationMode.BRONZE_AGE_COMPARISON,
        "teacher_max_epochs": 15,
        "train_concept_model": True,
        "student_layer_type": LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(
        experiment_title,
        [DatasetEnum.SIMPLE_SATURATION] + ALL_DATASETS,
        **specific_config,
    )


def run_student_teacher_training(
    student_layer_type: LayerTypeBronze, student_aggregation_mode: AggregationMode
):
    experiment_title = f"Student Teacher Training Stone Age to {student_aggregation_mode} on {student_layer_type}"
    specific_config = {
        "bounding_parameter": 10,
        "layer_type": LayerTypeBronze.MLP,
        "nonlinearity": NonLinearity.GUMBEL_SOFTMAX,
        "evaluation_nonlinearity": NonLinearity.DIFFERENTIABLE_ARGMAX,
        "concept_embedding_size": 256,
        "concept_temperature": 0.5,
        "entropy_loss_scaling": 0.0,
        "early_stopping": True,
        "loss_mode": LossMode.CROSS_ENTROPY,
        "train_decision_tree": False,
        "aggregation_mode": AggregationMode.STONE_AGE,
        "teacher_max_epochs": 30,
        "train_concept_model": True,
        "student_layer_type": student_layer_type,
        "student_aggregation_mode": student_aggregation_mode,
        "concept_memory_disjunctions": 4,
    }
    run_experiment(
        experiment_title,
        [DatasetEnum.SIMPLE_SATURATION] + ALL_DATASETS,
        **specific_config,
    )


def run_student_teacher_training_experiments():
    student_layer_types = [
        LayerTypeBronze.DEEP_CONCEPT_REASONER,
        LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER,
    ]
    student_aggregation_modes = [
        AggregationMode.BRONZE_AGE,
        AggregationMode.BRONZE_AGE_ROUNDED,
        AggregationMode.BRONZE_AGE_COMPARISON,
    ]

    for student_layer_type in student_layer_types:
        for student_aggregation_mode in student_aggregation_modes:
            run_student_teacher_training(student_layer_type, student_aggregation_mode)


run_dcr_ablation_study = lambda: run_ablation_study(
    LayerTypeBronze.DEEP_CONCEPT_REASONER
)
run_cmr_ablation_study = lambda: run_ablation_study(
    LayerTypeBronze.MEMORY_BASED_CONCEPT_REASONER
)

ablation_studies = [
    run_dcr_ablation_study,
    run_cmr_ablation_study,
]


base_tests = [run_stoneage, run_bronzeage_dcr, run_bronzeage_cmr]

explain_runs = [
    run_simple_saturation_stone_age,
    run_simple_saturation_bronze_age_dcr,
    run_simple_saturation_bronze_age_cmr,
]

if __name__ == "__main__":
    for exp in [run_student_teacher_training_experiments]:
        try:
            exp()
        except Exception as e:
            print(f"Experiment {exp.__name__} failed with error: {e}")
            import traceback

            traceback.print_exc()
            continue
    print("All experiments completed.")

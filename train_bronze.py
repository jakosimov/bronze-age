import logging
import warnings
from pathlib import Path

import lightning
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torchmetrics.classification import Accuracy

from bronze_age.config import AggregationMode
from bronze_age.config import BronzeConfig as Config
from bronze_age.config import DatasetEnum
from bronze_age.config import LayerTypeBronze as LayerType
from bronze_age.config import LossMode, NetworkType, NonLinearity
from bronze_age.datasets import DatasetEnum, get_dataset
from bronze_age.datasets.cross_validation import CrossValidationSplit
from bronze_age.models.bronze_age import BronzeAgeGNN
from bronze_age.models.decision_tree import train_decision_tree_model

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


from itertools import chain

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


def get_class_weights(
    train_dataset, validation_dataset, config, num_classes, device=None
):
    if config.dataset.uses_mask:
        y_train = train_dataset[0].y[train_dataset[0].train_mask].cpu().detach().numpy()
        y_val = (
            validation_dataset[0]
            .y[validation_dataset[0].val_mask]
            .cpu()
            .detach()
            .numpy()
        )
        y = np.concatenate([y_train, y_val])
    else:
        y = np.concatenate(
            [graph.y for graph in chain(train_dataset, validation_dataset)]
        )
    classes = np.unique(y)
    class_weights = np.ones(num_classes)
    computed_class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    )
    if len(computed_class_weights) < num_classes:
        for i in range(len(classes)):
            class_weights[classes[i]] = computed_class_weights[i]
    else:
        class_weights = computed_class_weights
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    if device is not None:
        class_weights = class_weights.to(device)
    return class_weights


def _binary_cross_entropy_loss(y_hat, y, class_weights):
    y_one_hot = F.one_hot(y.long(), num_classes=y_hat.shape[-1]).float()
    return F.binary_cross_entropy(y_hat, y_one_hot, weight=class_weights)


def _cross_entropy_loss(y_hat, y, class_weights):
    return F.cross_entropy(y_hat, y, weight=class_weights)


class LightningModel(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        config: Config,
        class_weights=None,
        suffix="",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.config = config
        self.model = model
        self.class_weights = class_weights
        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.suffix = suffix

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat, entropies, _ = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )

        y = batch.y
        if self.config.dataset.uses_mask:
            y_hat = y_hat[batch.train_mask]
            y = y[batch.train_mask]
        for key, value in entropies.items():
            self.log(
                f"key{self.suffix}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch.y.size(0),
            )
        entropy_loss = torch.stack(list(entropies.values())).sum()
        loss = (
            _binary_cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
            if self.config.loss_mode == LossMode.BINARY_CROSS_ENTROPY
            else _cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
        )
        final_loss = loss + self.config.entropy_loss_scaling * entropy_loss
        self.log(
            f"train_entropy_loss{self.suffix}", entropy_loss, batch_size=batch.y.size(0)
        )
        self.log(f"train_error_loss{self.suffix}", loss, batch_size=batch.y.size(0))
        self.log(f"train_loss{self.suffix}", final_loss, batch_size=batch.y.size(0))
        # print("final_loss", final_loss)

        self.train_accuracy(y_hat, y)
        self.log(
            f"train_acc{self.suffix}", self.train_accuracy, on_step=False, on_epoch=True
        )
        return final_loss

    def validation_step(self, batch, batch_idx):
        y_hat, entropies, _ = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        # NLL loss
        y = batch.y
        if self.config.dataset.uses_mask:
            y_hat = y_hat[batch.val_mask]
            y = y[batch.val_mask]
        loss = (
            _binary_cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
            if self.config.loss_mode == LossMode.BINARY_CROSS_ENTROPY
            else _cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
        )
        entropy_loss = torch.stack(list(entropies.values())).sum()
        loss = (
            _binary_cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
            if self.config.loss_mode == LossMode.BINARY_CROSS_ENTROPY
            else _cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
        )
        final_loss = loss + self.config.entropy_loss_scaling * entropy_loss

        self.log(
            f"val_loss{self.suffix}", final_loss, batch_size=y.size(0), on_epoch=True
        )

        self.val_accuracy(y_hat, y)
        self.log(
            f"val_acc{self.suffix}", self.val_accuracy, on_step=False, on_epoch=True
        )
        return final_loss

    def test_step(self, batch, batch_idx):
        y_hat, _, explanations = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            return_explanation=True,
        )
        explanation_path = (
            Path(self.loggers[0].log_dir) / f"explanations{self.suffix}.txt"
        )
        import json

        explanation_path.write_text(json.dumps(explanations, indent=4))
        # NLL loss
        y = batch.y
        if self.config.dataset.uses_mask:
            y_hat = y_hat[batch.test_mask]
            y = y[batch.test_mask]
        loss = (
            _binary_cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
            if self.config.loss_mode == LossMode.BINARY_CROSS_ENTROPY
            else _cross_entropy_loss(y_hat, y, self.class_weights.to(y_hat.device))
        )
        self.log(
            f"test_loss{self.suffix}", loss, batch_size=batch.y.size(0), on_epoch=True
        )

        self.val_accuracy(y_hat, y)
        self.log(
            f"test_acc{self.suffix}", self.val_accuracy, on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.learning_rate)


def train(config: Config):
    dataset = get_dataset(config)

    test_accuracies = []
    test_accuracies_dt = []
    test_accuracies_dt_pruned = []
    test_accuracies_cm = []

    start_time = pd.Timestamp.now().strftime("%d/%m/%y %H:%M")
    experiment_title = f"{start_time} {config.dataset} {config.layer_type}"

    split = CrossValidationSplit(config, dataset, random_state=42)
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(split):
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
        train_loader_test = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=100,
            verbose=False,
            mode="min",
            stopping_threshold=0.001,
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor="val_acc", mode="max"
        )

        class_weights = get_class_weights(
            train_dataset, val_dataset, config, dataset.num_classes, device=None
        )
        gnn = BronzeAgeGNN(dataset.num_node_features, dataset.num_classes, config)
        model = LightningModel(
            gnn,
            dataset.num_classes,
            config,
            class_weights=class_weights,
        )

        logger = pl_loggers.TensorBoardLogger(
            save_dir="lightning_logs", name=experiment_title, version=f"CV_{i+1}"
        )

        trainer = lightning.Trainer(
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            accelerator="cpu",
            callbacks=[checkpoint_callback]
            + ([early_stopping] if config.early_stopping else []),
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=logger,
        )
        trainer.fit(model, train_loader, val_dataloaders=val_loader)

        best_validation_model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=gnn,
        )

        best_train_accuracy = trainer.test(
            best_validation_model, train_loader_test, verbose=False
        )[0]["test_acc"]
        best_validation_accuracy = trainer.test(
            best_validation_model, val_loader, verbose=False
        )[0]["test_acc"]
        test_accuracy = trainer.test(best_validation_model, test_loader, verbose=False)[
            0
        ]["test_acc"]

        if config.train_decision_tree:
            model.eval()
            tree_model = best_validation_model.model.to_decision_tree(train_loader_test)
            wrapped_tree_model = LightningModel(
                tree_model,
                dataset.num_classes,
                config,
                class_weights=class_weights,
                suffix="_dt",
            )
            # save the decision tree model
            tree_model_path = (
                Path(logger.log_dir) / "checkpoints" / f"decision_tree_model.pt"
            )
            torch.save(tree_model, tree_model_path)
            test_accuracy_dt = trainer.test(
                wrapped_tree_model, test_loader, verbose=False
            )[0]["test_acc_dt"]
            test_accuracies_dt.append(test_accuracy_dt)

            # Pruning
            def score_model(model, loader):
                wrapped_model = LightningModel(
                    model,
                    dataset.num_classes,
                    config,
                    class_weights=class_weights,
                    suffix="_dt_while_pruning",
                )
                return trainer.test(wrapped_model, loader, verbose=False)[0][
                    "test_acc_dt_while_pruning"
                ]

            pruned_tree_model, num_nodes_pruned, num_nodes_remaining = (
                tree_model.prune_decision_trees(
                    train_loader_test, val_loader, score_model
                )
            )
            wrapped_pruned_model = LightningModel(
                pruned_tree_model,
                dataset.num_classes,
                config,
                class_weights=class_weights,
                suffix="_dt_pruned",
            )
            pruned_tree_model_path = (
                Path(logger.log_dir) / "checkpoints" / f"pruned_decision_tree_model.pt"
            )
            torch.save(pruned_tree_model, pruned_tree_model_path)
            test_accuracy_dt_pruned = trainer.test(
                wrapped_pruned_model, test_loader, verbose=False
            )[0]["test_acc_dt_pruned"]
            test_accuracies_dt_pruned.append(test_accuracy_dt_pruned)

        if config.train_concept_model:
            model.eval()
            concept_model = best_validation_model.model.train_concept_model(
                train_loader_test, experiment_title=experiment_title
            )
            wrapped_concept_model = LightningModel(
                concept_model,
                dataset.num_classes,
                config,
                class_weights=class_weights,
                suffix="_cm",
            )
            concept_model_path = (
                Path(logger.log_dir) / "checkpoints" / f"concept_model.pt"
            )
            torch.save(concept_model, concept_model_path)
            test_accuracy_cm = trainer.test(
                wrapped_concept_model, test_loader, verbose=False
            )[0]["test_acc_cm"]
            test_accuracies_cm.append(test_accuracy_cm)

        print(f"=====================")
        print(f"Fold {i+1}/{config.num_cv}")
        print(f"Train accuracy: {best_train_accuracy}")
        print(f"Validation accuracy: {best_validation_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        if config.train_decision_tree:
            print(f"Test accuracy DT: {test_accuracy_dt}")
            print(
                f"Test accuracy DT pruned ({num_nodes_remaining} nodes): {test_accuracy_dt_pruned}"
            )
        if config.train_concept_model:
            print(f"Test accuracy CM: {test_accuracy_cm}")
        print(f"=====================")

        test_accuracies.append(test_accuracy)
    print(f"=====================")
    print(f"Dataset {config.dataset}")
    print(
        f"Average test accuracy: {np.mean(test_accuracies)} with std: {np.std(test_accuracies)}"
    )
    if config.train_decision_tree:
        print(
            f"Average test accuracy DT: {np.mean(test_accuracies_dt)} with std: {np.std(test_accuracies_dt)}"
        )
        print(
            f"Average test accuracy DT pruned: {np.mean(test_accuracies_dt_pruned)} with std: {np.std(test_accuracies_dt_pruned)}"
        )
    if config.train_concept_model:
        print(
            f"Average test accuracy CM: {np.mean(test_accuracies_cm)} with std: {np.std(test_accuracies_cm)}"
        )
    print(f"=====================")

    return (
        np.mean(test_accuracies),
        np.std(test_accuracies),
        np.mean(test_accuracies_dt),
        np.std(test_accuracies_dt),
        np.mean(test_accuracies_dt_pruned),
        np.std(test_accuracies_dt_pruned),
        np.mean(test_accuracies_cm),
        np.std(test_accuracies_cm),
    )


def get_config_for_dataset(dataset, **kwargs):
    NUM_LAYERS = {
        DatasetEnum.INFECTION: 5,
        DatasetEnum.SATURATION: 1,
        DatasetEnum.SIMPLE_SATURATION: 1,
        DatasetEnum.BA_SHAPES: 5,
        DatasetEnum.TREE_CYCLE: 5,
        DatasetEnum.TREE_GRID: 5,
        DatasetEnum.BA_2MOTIFS: 4,
        DatasetEnum.MUTAG: 4,
        DatasetEnum.MUTAGENICITY: 3,
        DatasetEnum.BBBP: 3,
        DatasetEnum.PROTEINS: 3,
        DatasetEnum.IMDB_BINARY: 3,
        DatasetEnum.REDDIT_BINARY: 2,
        DatasetEnum.COLLAB: 3,
        DatasetEnum.SIMPLE_SATURATION: 1,
        DatasetEnum.DISTANCE: 1,
        DatasetEnum.PATH_FINDING: 1,
        DatasetEnum.PREFIX_SUM: 1,
        DatasetEnum.ROOT_VALUE: 1,
        DatasetEnum.GAME_OF_LIFE: 1,
        DatasetEnum.HEXAGONAL_GAME_OF_LIFE: 1,
    }
    NUM_STATES = {
        DatasetEnum.INFECTION: 6,
        DatasetEnum.SATURATION: 3,
        DatasetEnum.SIMPLE_SATURATION: 3,
        DatasetEnum.BA_SHAPES: 5,
        DatasetEnum.TREE_CYCLE: 5,
        DatasetEnum.TREE_GRID: 5,
        DatasetEnum.BA_2MOTIFS: 6,
        DatasetEnum.MUTAG: 6,
        DatasetEnum.MUTAGENICITY: 8,
        DatasetEnum.BBBP: 5,
        DatasetEnum.PROTEINS: 5,
        DatasetEnum.IMDB_BINARY: 5,
        DatasetEnum.REDDIT_BINARY: 5,
        DatasetEnum.COLLAB: 8,
        DatasetEnum.SIMPLE_SATURATION: 3,
        DatasetEnum.DISTANCE: 4,
        DatasetEnum.PATH_FINDING: 5,
        DatasetEnum.PREFIX_SUM: 6,
        DatasetEnum.ROOT_VALUE: 5,
        DatasetEnum.GAME_OF_LIFE: 2,
        DatasetEnum.HEXAGONAL_GAME_OF_LIFE: 2,
    }
    NUM_ITERATIONS = {
        DatasetEnum.DISTANCE: 12,
        DatasetEnum.PATH_FINDING: 12,
        DatasetEnum.PREFIX_SUM: 12,
        DatasetEnum.ROOT_VALUE: 3,
        DatasetEnum.GAME_OF_LIFE: 1,
        DatasetEnum.HEXAGONAL_GAME_OF_LIFE: 1,
    }
    BATCH_SIZES = {
        DatasetEnum.SATURATION: 1,
    }
    config = {
        "data_dir": "downloads",
        "temperature": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 10,
        "batch_size": BATCH_SIZES.get(dataset, 128),
        "learning_rate": 0.01,
        "max_epochs": 1500,
        "num_cv": 10,
        "dataset": dataset,
        "num_layers": NUM_LAYERS[dataset],
        "state_size": NUM_STATES[dataset],
        "layer_type": LayerType.MEMORY_BASED_CONCEPT_REASONER,
        "nonlinearity": None,
        "evaluation_nonlinearity": None,
        "concept_embedding_size": 128,
        "concept_temperature": 0.1,
        "entropy_loss_scaling": 0.2,
        "early_stopping": True,
        "loss_mode": LossMode.BINARY_CROSS_ENTROPY,
        "train_decision_tree": False,
        "aggregation_mode": AggregationMode.STONE_AGE,
        "num_recurrent_iterations": NUM_ITERATIONS.get(dataset, 1),
        "teacher_max_epochs": 15,
        "train_concept_model": False,
        "student_layer_type": LayerType.MEMORY_BASED_CONCEPT_REASONER,
        "student_aggregation_mode": AggregationMode.BRONZE_AGE_ROUNDED,
        "concept_memory_disjunctions": 4,
    }
    config.update(kwargs)
    return Config(**config)


def store_results(results, filename="results.csv", filename2="results2.csv"):
    df = pd.DataFrame(results).T
    df.columns = [
        "success",
        "mean_acc",
        "std_acc",
        "mean_acc_dt",
        "std_acc_dt",
        "mean_acc_dt_pruned",
        "std_acc_dt_pruned",
        "mean_acc_cm",
        "std_acc_cm",
    ]

    df.success = df.success.replace({True: "✅", False: "🛑"})
    df["uses_mask"] = df.index.map(lambda x: x.uses_mask)
    df["uses_pooling"] = df.index.map(lambda x: x.uses_pooling)
    df.to_csv(filename)

    df2 = df[["success", "uses_mask", "uses_pooling"]].copy()

    df2["GNN"] = (
        df["mean_acc"].map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc"].map("{:.2f}".format, na_action="ignore")
    )
    df2["DT"] = (
        df["mean_acc_dt"].fillna(-1).map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc_dt"].fillna(-1).map("{:.2f}".format, na_action="ignore")
    )
    df2.loc[df["mean_acc_dt"].isna(), "DT"] = "N/A"

    df2["DT pruned"] = (
        df["mean_acc_dt_pruned"].fillna(-1).map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc_dt_pruned"].fillna(-1).map("{:.2f}".format, na_action="ignore")
    )
    df2.loc[df["mean_acc_dt_pruned"].isna(), "DT pruned"] = "N/A"

    df2["CM"] = (
        df["mean_acc_cm"].fillna(-1).map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc_cm"].fillna(-1).map("{:.2f}".format, na_action="ignore")
    )
    df2.loc[df["mean_acc_cm"].isna(), "CM"] = "N/A"
    df2.to_csv(filename2)

    print(df2)


if __name__ == "__main__":
    results = {}
    datasets = [
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

    datasets = [DatasetEnum.SIMPLE_SATURATION, DatasetEnum.SATURATION]  # + datasets
    # datasets = [DatasetEnum.SIMPLE_SATURATION, DatasetEnum.INFECTION, DatasetEnum.MUTAG]
    # datasets = [DatasetEnum.BA_2MOTIFS]
    for dataset in datasets:
        for dataset_, (
            success,
            mean_acc,
            std_acc,
            mean_acc_dt,
            std_acc_dt,
            mean_acc_dt_pruned,
            std_acc_dt_pruned,
            mean_acc_cm,
            std_acc_cm,
        ) in results.items():
            store_results(
                results, filename="results_temp.csv", filename2="results2_temp.csv"
            )
            if success:
                print(
                    f"✅ {dataset_}: GNN {mean_acc:.2f} ± {std_acc:.2f}, DT {mean_acc_dt:.2f} ± {std_acc_dt:.2f}, DT pruned {mean_acc_dt_pruned:.2f} ± {std_acc_dt_pruned:.2f}, CM {mean_acc_cm:.2f} ± {std_acc_cm:.2f}"
                )
            else:
                print(f"🛑 {dataset_}")
        print("Running dataset:", dataset)
        try:
            config = get_config_for_dataset(dataset)
            lightning.seed_everything(0, workers=True)
            (
                mean_acc,
                std_acc,
                mean_acc_dt,
                std_acc_dt,
                mean_acc_dt_pruned,
                std_acc_dt_pruned,
                mean_acc_cm,
                std_acc_cm,
            ) = train(config)
            results[dataset] = (
                True,
                mean_acc,
                std_acc,
                mean_acc_dt,
                std_acc_dt,
                mean_acc_dt_pruned,
                std_acc_dt_pruned,
                mean_acc_cm,
                std_acc_cm,
            )
        except Exception as e:
            print(f"Error with dataset {dataset}: {e}")
            results[dataset] = (False, None, None, None, None, None, None, None, None)
            import traceback

            traceback.print_exc()
            # raise e

    for dataset_, (
        success,
        mean_acc,
        std_acc,
        mean_acc_dt,
        std_acc_dt,
        mean_acc_dt_pruned,
        std_acc_dt_pruned,
        mean_acc_cm,
        std_acc_cm,
    ) in results.items():
        if success:
            print(
                f"✅ {dataset_}: GNN {mean_acc:.2f} ± {std_acc:.2f}, DT {mean_acc_dt:.2f} ± {std_acc_dt:.2f}, DT pruned {mean_acc_dt_pruned:.2f} ± {std_acc_dt_pruned:.2f}, CM {mean_acc_cm:.2f} ± {std_acc_cm:.2f}"
            )
        else:
            print(f"🛑 {dataset_}")

    store_results(results, filename="results.csv", filename2="results2.csv")

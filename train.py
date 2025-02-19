import logging
import warnings

import lightning
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torchmetrics.classification import Accuracy

from bronze_age.config import Config, NetworkType
from bronze_age.datasets import DatasetEnum, get_dataset
from bronze_age.models.decision_tree import train_decision_tree_model
from bronze_age.models.stone_age import StoneAgeGNN as BronzeAgeGNN

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


def get_class_weights(y, num_classes, device=None):
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


def train(config: Config):
    dataset = get_dataset(config)

    class LightningModel(lightning.LightningModule):
        def __init__(self, class_weights=None):
            super().__init__()
            self.model = BronzeAgeGNN(
                dataset.num_node_features, dataset.num_classes, config
            )
            self.class_weights = class_weights
            self.train_accuracy = Accuracy(
                task="multiclass", num_classes=dataset.num_classes, average="micro"
            )
            self.val_accuracy = Accuracy(
                task="multiclass", num_classes=dataset.num_classes, average="micro"
            )
            self.test_accuracy = Accuracy(
                task="multiclass", num_classes=dataset.num_classes, average="micro"
            )

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            y_hat = self.model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch
            )

            y = batch.y
            if config.dataset.uses_mask:
                y_hat = y_hat[batch.train_mask]
                y = y[batch.train_mask]
            loss = F.nll_loss(y_hat, y, weight=self.class_weights)
            self.log("train_loss", loss, batch_size=batch.y.size(0))

            self.train_accuracy(y_hat, y)
            self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            y_hat = self.model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch
            )
            # NLL loss
            y = batch.y
            if config.dataset.uses_mask:
                y_hat = y_hat[batch.val_mask]
                y = y[batch.val_mask]
            loss = F.nll_loss(y_hat, y, weight=self.class_weights)
            self.log("val_loss", loss, batch_size=y.size(0), on_epoch=True)

            self.val_accuracy(y_hat, y)
            self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
            return loss

        def test_step(self, batch, batch_idx):
            y_hat = self.model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch
            )
            # NLL loss
            y = batch.y
            if config.dataset.uses_mask:
                y_hat = y_hat[batch.test_mask]
                y = y[batch.test_mask]
            loss = F.nll_loss(y_hat, y, weight=self.class_weights)
            self.log("test_loss", loss, batch_size=batch.y.size(0), on_epoch=True)

            self.val_accuracy(y_hat, y)
            self.log("test_acc", self.val_accuracy, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    class LightningTestWrapper(lightning.LightningModule):
        def __init__(self, model, class_weights=None):
            super().__init__()
            self.model = model
            self.test_accuracy = Accuracy(
                task="multiclass", num_classes=dataset.num_classes, average="micro"
            )
            self.class_weights = class_weights

        def forward(self, x):
            return self.model(x)

        def test_step(self, batch, batch_idx):
            y_hat = self.model(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch
            )
            # NLL loss
            y = batch.y
            if config.dataset.uses_mask:
                y_hat = y_hat[batch.test_mask]
                y = y[batch.test_mask]
            loss = F.nll_loss(y_hat, y, weight=self.class_weights)
            self.log("test_loss_dt", loss, batch_size=batch.y.size(0), on_epoch=True)

            self.test_accuracy(y_hat, y)
            self.log("test_acc_dt", self.test_accuracy, on_step=False, on_epoch=True)
            return loss

    skf = StratifiedKFold(n_splits=config.num_cv, shuffle=True, random_state=42)
    if config.dataset.uses_pooling:
        labels = [graph.y[0] for graph in dataset]
    elif config.dataset.uses_mask:
        if len(dataset) == 1:
            labels = dataset[0].y
        else:
            raise NotImplementedError("Masking not implemented for multiple graphs")
    else:
        # no pooling and no masking ->
        # take random graphs from dataset
        labels = [0 for _ in dataset]

    test_accuracies = []
    test_accuracies_dt = []
    for i, (train_index, test_index) in enumerate(
        skf.split([0 for _ in labels], labels)
    ):
        sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=41)

        if not config.dataset.uses_mask:
            # print(len(train_val_dataset))
            train_val_dataset = dataset[train_index]
            X = [data.x for data in train_val_dataset]
            y = [data.y for data in train_val_dataset]
            train_index, val_index = next(sss.split(X, y))

            train_dataset = train_val_dataset[train_index]
            val_dataset = train_val_dataset[val_index]
            test_dataset = dataset[test_index]

            y = np.concatenate(y)
        else:
            if len(dataset) == 1:
                data = dataset[0].clone()
                X = data.x[train_index].cpu().detach().numpy()
                y = data.y[train_index].cpu().detach().numpy()
                train_index_, val_index_ = next(sss.split(X, y))
                train_mask = train_index[train_index_]
                val_mask = train_index[val_index_]
                test_mask = test_index
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask
                train_dataset = [data]
                val_dataset = [data]
                test_dataset = [data]
            else:
                raise NotImplementedError("Masking not implemented for multiple graphs")

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=100, verbose=False, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor="val_loss", mode="min"
        )

        class_weights = get_class_weights(y, dataset.num_classes, device=None)
        model = LightningModel(class_weights=class_weights)

        trainer = lightning.Trainer(
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            accelerator="cpu",
            callbacks=[early_stopping, checkpoint_callback],
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_dataloaders=val_loader)

        best_validation_model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        best_train_accuracy = trainer.test(
            best_validation_model, train_loader, verbose=False
        )[0]["test_acc"]
        best_validation_accuracy = trainer.test(
            best_validation_model, val_loader, verbose=False
        )[0]["test_acc"]
        test_accuracy = trainer.test(best_validation_model, test_loader, verbose=False)[
            0
        ]["test_acc"]

        tree_model = train_decision_tree_model(
            model.model, config, dataset.num_classes, train_dataset, val_dataset
        )
        wrapped_tree_model = LightningTestWrapper(tree_model)
        test_accuracy_dt = trainer.test(wrapped_tree_model, test_loader, verbose=False)[
            0
        ]["test_acc_dt"]
        test_accuracies_dt.append(test_accuracy_dt)

        print(f"=====================")
        print(f"Fold {i+1}/{config.num_cv}")
        print(f"Train accuracy: {best_train_accuracy}")
        print(f"Validation accuracy: {best_validation_accuracy}")
        print(f"Test accuracy: {test_accuracy}")
        print(f"Test accuracy DT: {test_accuracy_dt}")
        print(f"=====================")

        test_accuracies.append(test_accuracy)
    print(f"=====================")
    print(f"Dataset {config.dataset}")
    print(
        f"Average test accuracy: {np.mean(test_accuracies)} with std: {np.std(test_accuracies)}"
    )
    print(
        f"Average test accuracy DT: {np.mean(test_accuracies_dt)} with std: {np.std(test_accuracies_dt)}"
    )
    print(f"=====================")

    return (
        np.mean(test_accuracies),
        np.std(test_accuracies),
        np.mean(test_accuracies_dt),
        np.std(test_accuracies_dt),
    )


def get_config_for_dataset(dataset, **kwargs):
    NUM_LAYERS = {
        DatasetEnum.INFECTION: 5,
        DatasetEnum.SATURATION: 1,
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
    }
    NUM_STATES = {
        DatasetEnum.INFECTION: 6,
        DatasetEnum.SATURATION: 3,
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
    }
    config = {
        "data_dir": "downloads",
        "temperature": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "dropout": 0.0,
        "use_batch_norm": True,
        "network": NetworkType.MLP,
        "hidden_units": 16,
        "skip_connection": True,
        "bounding_parameter": 1000,
        "batch_size": 128,
        "learning_rate": 0.01,
        "max_epochs": 1500,
        "num_cv": 10,
        "dataset": dataset,
        "num_layers": NUM_LAYERS[dataset],
        "state_size": NUM_STATES[dataset],
    }
    config.update(kwargs)
    return Config(**config)


def store_results(results):
    df = pd.DataFrame(results).T
    df.columns = ["success", "mean_acc", "std_acc", "mean_acc_dt", "std_acc_dt"]

    df.success = df.success.replace({True: "✅", False: "🛑"})
    df2 = df[["success"]]

    df2["GNN"] = (
        df["mean_acc"].map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc"].map("{:.2f}".format, na_action="ignore")
    )
    df2["DT"] = (
        df["mean_acc_dt"].map("{:.2f}".format, na_action="ignore")
        + " ± "
        + df["std_acc_dt"].map("{:.2f}".format, na_action="ignore")
    )
    df.to_csv("results.csv")
    df2.to_csv("results2.csv")

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
        DatasetEnum.MUTAG,
        DatasetEnum.MUTAGENICITY,
        DatasetEnum.BBBP,
        DatasetEnum.PROTEINS,
        DatasetEnum.IMDB_BINARY,
        DatasetEnum.REDDIT_BINARY,
        DatasetEnum.COLLAB,
    ]

    for dataset in datasets:
        for dataset_, (
            success,
            mean_acc,
            std_acc,
            mean_acc_dt,
            std_acc_dt,
        ) in results.items():
            if success:
                print(
                    f"✅ {dataset_}: GNN {mean_acc:.2f} ± {std_acc:.2f}, DT {mean_acc_dt:.2f} ± {std_acc_dt:.2f}"
                )
            else:
                print(f"🛑 {dataset_}")
        print("Running dataset:", dataset)
        try:
            config = get_config_for_dataset(dataset)
            lightning.seed_everything(0)
            mean_acc, std_acc, mean_acc_dt, std_acc_dt = train(config)
            results[dataset] = (True, mean_acc, std_acc, mean_acc_dt, std_acc_dt)
        except Exception as e:
            print(f"Error with dataset {dataset}: {e}")
            results[dataset] = (False, None, None, None, None)
            raise e

    for dataset_, (
        success,
        mean_acc,
        std_acc,
        mean_acc_dt,
        std_acc_dt,
    ) in results.items():
        if success:
            print(
                f"✅ {dataset_}: GNN {mean_acc:.2f} ± {std_acc:.2f}, DT {mean_acc_dt:.2f} ± {std_acc_dt:.2f}"
            )
        else:
            print(f"🛑 {dataset_}")

    store_results(results)

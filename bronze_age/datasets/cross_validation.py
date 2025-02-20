import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

from bronze_age.config import Config


class CrossValidationSplit:
    def __init__(self, config: Config, dataset, random_state=42):
        self.config = config
        self.dataset = dataset
        self.random_state = random_state

    def __iter__(self):
        config = self.config
        dataset = self.dataset
        random_state = self.random_state
        skf = StratifiedKFold(
            n_splits=config.num_cv, shuffle=True, random_state=random_state
        )
        if config.dataset.uses_pooling:
            labels = [graph.y[0] for graph in dataset]
        elif config.dataset.uses_mask:
            if len(dataset) == 1:
                labels = dataset[0].y
            else:
                raise NotImplementedError("Masking not implemented for multiple graphs")
        else:
            # no pooling and no masking -> take random graphs from dataset
            labels = [0 for _ in dataset]

        for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
            random_state += 1
            sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)

            if not config.dataset.uses_mask:
                # Split the full dataset
                train_val_dataset = dataset[train_index]
                X = [data.x for data in train_val_dataset]
                y = [data.y for data in train_val_dataset]
            elif len(dataset) == 1:
                # Split the nodes
                train_val_dataset = dataset[0].clone()
                X = train_val_dataset.x[train_index].cpu().detach().numpy()
                y = train_val_dataset.y[train_index].cpu().detach().numpy()
            else:
                raise NotImplementedError("Masking not implemented for multiple graphs")

            train_index_, val_index_ = next(sss.split(X, y))

            if not config.dataset.uses_mask:
                # Split the full dataset
                train_dataset = train_val_dataset[train_index_]
                val_dataset = train_val_dataset[val_index_]
                test_dataset = dataset[test_index]

                y = np.concatenate(y)
            else:
                # Split the nodes of the only graph
                train_mask = train_index[train_index_]
                val_mask = train_index[val_index_]
                test_mask = test_index
                train_val_dataset.train_mask = train_mask
                train_val_dataset.val_mask = val_mask
                train_val_dataset.test_mask = test_mask
                train_dataset = [train_val_dataset]
                val_dataset = [train_val_dataset]
                test_dataset = [train_val_dataset]

            yield train_dataset, val_dataset, test_dataset

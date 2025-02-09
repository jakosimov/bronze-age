
import lightning
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader

from bronze_age.config import Config
from bronze_age.datasets import DatasetEnum, get_dataset
from bronze_age.models.stone_age import StoneAgeGNN


def train(config: Config):
    dataset = get_dataset(config)
    
    config.use_pooling = config.dataset.uses_pooling
    config.use_mask = config.dataset.uses_mask # TODO
    
    class LightningModel(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = StoneAgeGNN(dataset.num_node_features, dataset.num_classes, config)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            y_hat = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(y_hat, batch.y)
            self.log('train_loss', loss, batch_size=batch.y.size(0))
            train_acc = (y_hat.argmax(dim=1) == batch.y).sum().item() / batch.y.size(0)
            self.log('train_acc', train_acc, batch_size=batch.y.size(0))
            return loss

        def validation_step(self, batch, batch_idx):
            y_hat = self.model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(y_hat, batch.y)
            self.log('val_loss', loss)
            val_acc = (y_hat.argmax(dim=1) == batch.y).sum().item() / batch.y.size(0)
            self.log('val_acc', val_acc)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)
        
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    labels = [graph.y[0] for graph in dataset]
    if not config.use_pooling:
        labels = [0 for _ in dataset]
    for train_index, test_index in skf.split([0 for _ in labels], labels):
        train_loader = DataLoader([dataset[i] for i in train_index], batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader([dataset[i] for i in test_index], batch_size=config.batch_size, shuffle=False)
        model = LightningModel()
        trainer = lightning.Trainer(max_epochs=1500, log_every_n_steps=1)
        trainer.fit(model, train_loader, val_dataloaders=test_loader)
        final_accuracy = trainer.logged_metrics['val_acc']
        print(f"Final accuracy: {final_accuracy}")

if __name__ == '__main__':
    config = Config(dataset=DatasetEnum.MUTAG, data_dir='downloads', num_layers=4, state_size=6)
    train(config)
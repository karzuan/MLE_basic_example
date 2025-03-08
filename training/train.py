"""
This script prepares the data, runs the training, and saves the model.
"""
from datetime import datetime
from utils import get_project_dir, configure_logging
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchmetrics import Accuracy
import argparse
import os
import sys
import pickle
import json
import logging
import time
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
# CONF_FILE = os.getenv('CONF_PATH')
if os.path.exists("../settings.json"):
    CONF_FILE = "../settings.json"
else:
    CONF_FILE = "settings.json"

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")


class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class IrisModel(pl.LightningModule):
    def __init__(self, input_dim=4, output_dim=3, learning_rate=0.01):
        super(IrisModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5),
            nn.Linear(8, 16),
            nn.Sigmoid(),
            nn.Linear(16, output_dim)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        acc = self.accuracy(y_pred, y_batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': acc}

    def test_step(self, batch, batch_idx):  # Added test_step method
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        acc = self.accuracy(y_pred, y_batch)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Training:
    def __init__(self) -> None:
        self.model = IrisModel()

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        x_train, x_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(x_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(x_test, y_test)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        return train_test_split(df[['sepal length (cm)',
                                    'sepal width (cm)',
                                    'petal length (cm)',
                                    'petal width (cm)']],
                                df['y'], test_size=test_size, random_state=conf['general']['random_state'])

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        x_train = torch.tensor(x_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(self.model, train_loader)

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> list:
        logging.info("Testing the model...")
        x_test = torch.tensor(x_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)
        trainer = pl.Trainer()
        res = trainer.test(self.model, test_loader)
        # y_pred = self.model.predict(x_test)
        # res = f1_score(y_test, y_pred)
        logging.info(f"test results: {res}")
        return res

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            print(f"Created directory: {MODEL_DIR}")
            logging.info(f"Created directory: {MODEL_DIR}")
        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
            print(f"Model saved at: {path}")
            logging.info(f"Model saved at: {path}")
        else:
            path = os.path.join(MODEL_DIR, path)
            logging.info(f"Model saved at: {path}")

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()

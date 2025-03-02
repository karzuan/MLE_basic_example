"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle # library for serializing and de-serializing Python objects
import sys
from datetime import datetime
from utils import get_project_dir, configure_logging
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
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
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class IrisModel(pl.LightningModule):
    def __init__(self, input_dim=4, output_dim=3, learning_rate=0.01):
        # input_dim: number of features in the input
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
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=3)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step method is called for each batch
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

    def test_step(self, batch, batch_idx): # Added test_step method
        x_batch, y_batch = batch
        y_pred = self(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        acc = self.accuracy(y_pred, y_batch)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'accuracy': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    logging.info(f'Latest model: {latest}')
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str):
    """Loads and returns the specified model"""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            logging.info(f'Path of the model: {path}')
            return model
    except Exception as e:
        logging.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """loads and returns data for inference from the specified csv file"""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: IrisModel, infer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""
    results = model.predict(infer_data)
    infer_data['results'] = results
    return infer_data


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main function"""
    configure_logging()
    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model, infer_data)
    store_results(results, args.out_path)

    logging.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()

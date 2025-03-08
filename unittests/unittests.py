import unittest
import pandas as pd
import os
import sys
import json
import torch
from training.train import DataProcessor
from training.train import Training

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)
# print('root dir', ROOT_DIR)
sys.path.append(os.path.dirname(ROOT_DIR))
# CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = ROOT_DIR + '/settings.json'
# CONF_FILE = "../settings.json"
# print(f"Current working directory: {os.getcwd()}")


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = ROOT_DIR + '/' + conf['general']['data_dir']
        print(f"Data directory: {cls.data_dir}")
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_data_extraction(self):
        dp = DataProcessor()
        print(f"Train directory: {self.train_path}")
        df = dp.data_extraction(self.train_path)
        print(f"Dataframe shape: {df.shape}")
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data()
        self.assertEqual(df.shape[0], 120)


class TestTraining(unittest.TestCase):
    def test_train(self):
        tr = Training()
        # assume you have some prepared data
        x_train = pd.DataFrame({
            'sepal length(cm)': [4.6, 5.7, 6.4, 4.8],
            'sepal width(cm)': [3.5, 4.4, 3.3, 2.2],
            'petal length(cm)': [1.0, 1.5, 4.4, 1.5],
            'petal width(cm)': [0.3, 0.4, 1.3, 0.2]
        })
        y_train = pd.Series([0, 0, 1, 0])
        tr.train(x_train, y_train)
        self.assertIsInstance(tr.model, torch.nn.Module)


if __name__ == '__main__':
    unittest.main()

from dataset import TestDataset
import torch
import torch.nn as nn
import pandas as pd
from glob import glob


import sys
import easydict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("informer")
from informer.models.model import Informer


class Predictor:
    def __init__(self, sample_csv, dir_path):
        self.ans = pd.read_csv(sample_csv)
        self.csv_list = sorted(glob(dir_path))
        self.scaler = {}

    def predict(self, args, model, breed, path):
        column_name = breed + "_가격(원/kg)"
        target_column = self.ans.loc[:, column_name].tolist()
        for i, data in enumerate(self.dataset_list):
            price = data["price_scaled"].unsqueeze(0).unsqueeze(2)
            volume = data["volume_scaled"].unsqueeze(0).unsqueeze(2)
            x_mark = data["x_mark"].unsqueeze(0).permute(0, 2, 1).to(args.device)
            y_mark = data["y_mark"].unsqueeze(0).permute(0, 2, 1).to(args.device)
            y = data["y"].unsqueeze(0).unsqueeze(2)
            x = torch.cat([price, volume], dim=2).to(args.device)
            dec_inp = torch.zeros([y.shape[0], args.target_window, 1])
            y = torch.cat([y[:, : args.label_len, :], dec_inp], dim=1).to(args.device)

            outputs = model(x, x_mark, y, y_mark).cpu().detach().numpy()

            price_scaler = self.scaler[breed]
            
            pred = price_scaler.inverse_transform(outputs)
            pred = pred.squeeze(2).squeeze(0)
            target_column[3 * i] = pred[6]
            target_column[3 * i + 1] = pred[13]
            target_column[3 * i + 2] = pred[27]
        self.ans[column_name] = target_column
        self.save(path)

    def get_dataset(self, args, breed, price_scaler, volume_scaler):
        self.dataset_list = []
        self.scaler[breed] = price_scaler
        for csv in self.csv_list:
            self.dataset_list.append(
                TestDataset(
                    csv,
                    breed,
                    args.input_window,
                    args.target_window,
                    args.label_len,
                    price_scaler,
                    volume_scaler,
                )[0]
            )

    def save(self, path):
        self.ans.to_csv(path, index=False)

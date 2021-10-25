import csv
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import easydict


class TestDataset(Dataset):
    def __init__(
        self,
        dirname,
        breed,
        input_window,
        target_window,
        label_len,
        price_scaler,
        volume_scaler,
    ):
        f = open(dirname, "r", encoding="utf-8")  # file open
        rdr = csv.reader(f)  # csv read

        # 변수들을 저장한다
        self.breed_amount_name = breed + "_거래량(kg)"  # dictionary name for 거래량
        self.breed_price_name = breed + "_가격(원/kg)"  # dictionary name for 가격
        self.input_window = input_window  # window size(list size for 거래량 and 가격)
        self.target_window = target_window

        self.volume_scaled = []  # list for 거래량
        self.price_scaled = []  # list for 가격
        self.year = []
        self.month = []
        self.day = []
        self.weekday = []
        self.y = []  # label_len + ground truth
        self.y_mark = []
        self.x_mark = []

        self.price_scaler = price_scaler
        self.volume_scaler = volume_scaler

        volume_list = []
        price_list = []
        date_list = []
        weekday_list = []
        itr = -1
        for line in rdr:
            if itr == -1:
                name_list = line  # 이름이 담긴 첫번째 list 들을 뽑는다.
                # list에서 원하는 이름에 해당하는 index 값을 찾는다.
                volume_index = name_list.index(self.breed_amount_name)
                price_index = name_list.index(self.breed_price_name)
                date_index = name_list.index("date")
                day_index = name_list.index("요일")

            if itr >= 0:  # index에 해당하는 값을 가지고 list에 값을 append 한다.
                volume = line[volume_index]
                price = line[price_index]
                date = line[date_index]
                day = line[day_index]

                volume_list.append(volume)
                price_list.append(price)
                date_list.append(date)
                weekday_list.append(day)
            itr += 1
        f.close()
        volume_list = linear(volume_list)
        price_list = linear(price_list)

        volume_list = np.array(volume_list, dtype=np.float32)
        price_list = np.array(price_list, dtype=np.float32)
        weekday_list = np.array(weekday_list)

        self.volume_scaler.fit(volume_list.reshape(-1, 1))
        self.price_scaler.fit(price_list.reshape(-1, 1))

        # list scaling
        volume_list_scaled = np.squeeze(
            self.volume_scaler.transform(volume_list.reshape(-1, 1)).T
        )
        price_list_scaled = np.squeeze(
            self.price_scaler.transform(price_list.reshape(-1, 1)).T
        )
        volume_scaled = np.array(volume_list_scaled)
        price_scaled = np.array(price_list_scaled)

        year_list = []
        month_list = []
        day_list = []

        for i in range(1, len(date_list) + 1):
            temp_list = date_list[i - 1].split("-")
            year_element = int(temp_list[0]) - 2014
            month_element = int(temp_list[1])
            day_element = int(temp_list[2])
            year_list.append(year_element)
            month_list.append(month_element)
            day_list.append(day_element)

        year_list = np.array(year_list)
        month_list = np.array(month_list)
        day_list = np.array(day_list)

        ### Column 순서
        df = pd.DataFrame(
            [
                price_scaled,
                volume_scaled,
                year_list,
                month_list,
                day_list,
                weekday_list,
            ]
        ).T

        ### 요일을 숫자로 변환
        weekdays = {
            "일요일": 0,
            "월요일": 1,
            "화요일": 2,
            "수요일": 3,
            "목요일": 4,
            "금요일": 5,
            "토요일": 6,
        }
        df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: weekdays[x])

        self.price_scaled.append(df.iloc[: self.input_window, 0].tolist())
        self.volume_scaled.append(df.iloc[: self.input_window, 1].tolist())
        self.year.append(df.iloc[: self.input_window, 2].tolist())
        self.month.append(df.iloc[: self.input_window, 3].tolist())
        self.day.append(df.iloc[: self.input_window, 4].tolist())
        self.weekday.append(df.iloc[: self.input_window, 5].tolist())
        self.y.append(
            df.iloc[
                self.input_window - label_len : self.input_window + self.target_window,
                0,
            ].tolist()
        )
        self.x_mark.append(
            [
                df.iloc[: self.input_window, 5].tolist(),
                df.iloc[: self.input_window, 2].tolist(),
                df.iloc[: self.input_window, 3].tolist(),
                df.iloc[: self.input_window, 4].tolist(),
            ]
        )
        self.y_mark.append(
            [
                df.iloc[
                    self.input_window
                    - label_len : self.input_window
                    + self.target_window,
                    5,
                ].tolist(),
                df.iloc[
                    self.input_window
                    - label_len : self.input_window
                    + self.target_window,
                    2,
                ].tolist(),
                df.iloc[
                    self.input_window
                    - label_len : self.input_window
                    + self.target_window,
                    3,
                ].tolist(),
                df.iloc[
                    self.input_window
                    - label_len : self.input_window
                    + self.target_window,
                    4,
                ].tolist(),
            ]
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "price_scaled": torch.tensor(self.price_scaled[index]),
            "volume_scaled": torch.tensor(self.volume_scaled[index]),
            "year": torch.tensor(self.year[index]),
            "month": torch.tensor(self.month[index]),
            "day": torch.tensor(self.day[index]),
            "weekday": torch.tensor(self.weekday[index]),
            "y": torch.tensor(self.y[index]),
            "x_mark": torch.tensor(self.x_mark[index]),
            "y_mark": torch.tensor(self.y_mark[index]),
        }


class CustomDataset(Dataset):
    def __init__(self, dirname, breed, input_window, target_window, label_len):

        # csv 파일을 읽는다
        f = open(dirname, "r", encoding="utf-8")  # file open
        rdr = csv.reader(f)  # csv read

        # 변수들을 저장한다
        self.breed_amount_name = breed + "_거래량(kg)"  # dictionary name for 거래량
        self.breed_price_name = breed + "_가격(원/kg)"  # dictionary name for 가격
        self.input_window = input_window  # window size(list size for 거래량 and 가격)
        self.target_window = target_window

        self.volume = []  # list for 거래량
        self.price = []  # list for 가격

        self.date = []
        self.year = []
        self.month = []
        self.day = []
        self.weekday = []
        self.y = []  # label_len + ground truth
        self.y_mark = []
        self.x_mark = []

        # min max scaled
        self.volume_min_max = []
        self.price_min_max = []

        # stadard scaled
        self.volume_std = []
        self.price_std = []

        # standard, normalize 함수 정의
        self.mm_scaler_volume = MinMaxScaler()
        self.mm_scaler = MinMaxScaler()
        self.std_scaler_volume = StandardScaler()
        self.std_scaler = StandardScaler()

        # breed에 해당하는 거래량과 가격을 읽어 임시 list를 만든다.
        volume_list = []
        price_list = []
        date_list = []
        weekday_list = []
        itr = -1
        for line in rdr:
            if itr == -1:
                name_list = line  # 이름이 담긴 첫번째 list 들을 뽑는다.
                # list에서 원하는 이름에 해당하는 index 값을 찾는다.
                volume_index = name_list.index(self.breed_amount_name)
                price_index = name_list.index(self.breed_price_name)
                date_index = name_list.index("date")
                day_index = name_list.index("요일")

            if itr >= 0:  # index에 해당하는 값을 가지고 list에 값을 append 한다.
                volume = line[volume_index]
                price = line[price_index]
                date = line[date_index]
                day = line[day_index]

                volume_list.append(volume)
                price_list.append(price)
                date_list.append(date)
                weekday_list.append(day)
            itr += 1
        f.close()
        volume_list = linear(volume_list)
        price_list = linear(price_list)
        volume_list = np.array(volume_list, dtype=np.float32)
        price_list = np.array(price_list, dtype=np.float32)
        date_list = np.array(date_list)
        weekday_list = np.array(weekday_list)

        # scaler fitting
        self.mm_scaler_volume.fit(volume_list.reshape(-1, 1))
        self.mm_scaler.fit(price_list.reshape(-1, 1))
        self.std_scaler_volume.fit(volume_list.reshape(-1, 1))
        self.std_scaler.fit(price_list.reshape(-1, 1))

        # list scaling
        volume_list_min_max_scaled = np.squeeze(
            self.mm_scaler_volume.transform(volume_list.reshape(-1, 1)).T
        )
        price_list_min_max_scaled = np.squeeze(
            self.mm_scaler.transform(price_list.reshape(-1, 1)).T
        )
        volume_list_std_scaled = np.squeeze(
            self.std_scaler_volume.transform(volume_list.reshape(-1, 1)).T
        )
        price_list_std_scaled = np.squeeze(
            self.std_scaler.transform(price_list.reshape(-1, 1)).T
        )

        volume_list_min_max_scaled = np.array(volume_list_min_max_scaled)
        price_list_min_max_scaled = np.array(price_list_min_max_scaled)

        volume_list_std_scaled = np.array(volume_list_std_scaled)
        price_list_std_scaled = np.array(price_list_std_scaled)

        # 년, 월, 일 리스트 만드는 코드

        year_list = []
        month_list = []
        day_list = []

        for i in range(1, len(date_list) + 1):
            temp_list = date_list[i - 1].split("-")
            year_element = int(temp_list[0]) - 2014
            month_element = int(temp_list[1])
            day_element = int(temp_list[2])
            year_list.append(year_element)
            month_list.append(month_element)
            day_list.append(day_element)

        year_list = np.array(year_list)
        month_list = np.array(month_list)
        day_list = np.array(day_list)

        assert volume_list.shape == price_list.shape
        assert date_list.shape == weekday_list.shape

        ### Column 순서
        df = pd.DataFrame(
            [
                price_list,
                volume_list,
                price_list_min_max_scaled,
                volume_list_min_max_scaled,
                price_list_std_scaled,
                volume_list_std_scaled,
                date_list,
                year_list,
                month_list,
                day_list,
                weekday_list,
            ]
        ).T

        ### 요일을 숫자로 변환
        weekdays = {
            "일요일": 0,
            "월요일": 1,
            "화요일": 2,
            "수요일": 3,
            "목요일": 4,
            "금요일": 5,
            "토요일": 6,
        }
        df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: weekdays[x])

        for i in range(len(df) - self.target_window - self.input_window):
            self.price.append(df.iloc[i : i + self.input_window, 0].tolist())
            self.volume.append(df.iloc[i : i + self.input_window, 1].tolist())
            self.price_min_max.append(df.iloc[i : i + self.input_window, 2].tolist())
            self.volume_min_max.append(df.iloc[i : i + self.input_window, 3].tolist())
            self.price_std.append(df.iloc[i : i + self.input_window, 4].tolist())
            self.volume_std.append(df.iloc[i : i + self.input_window, 5].tolist())
            self.date.append(df.iloc[i : i + self.input_window, 6].tolist())
            self.year.append(df.iloc[i : i + self.input_window, 7].tolist())
            self.month.append(df.iloc[i : i + self.input_window, 8].tolist())
            self.day.append(df.iloc[i : i + self.input_window, 9].tolist())
            self.weekday.append(df.iloc[i : i + self.input_window, 10].tolist())
            self.y.append(
                df.iloc[
                    i
                    + self.input_window
                    - label_len : i
                    + self.input_window
                    + self.target_window,
                    4, ############ price_std
                ].tolist()
            )
            self.x_mark.append(
                [
                    df.iloc[
                        i : i + self.input_window,
                        10,
                    ].tolist(),
                    df.iloc[i : i + self.input_window, 7].tolist(),
                    df.iloc[i : i + self.input_window, 8].tolist(),
                    df.iloc[i : i + self.input_window, 9].tolist(),
                ]
            )
            self.y_mark.append(
                [
                    df.iloc[
                        i
                        + self.input_window
                        - label_len : i
                        + self.input_window
                        + target_window,
                        10,
                    ].tolist(),
                    df.iloc[
                        i
                        + self.input_window
                        - label_len : i
                        + self.input_window
                        + target_window,
                        7,
                    ].tolist(),
                    df.iloc[
                        i
                        + self.input_window
                        - label_len : i
                        + self.input_window
                        + target_window,
                        8,
                    ].tolist(),
                    df.iloc[
                        i
                        + self.input_window
                        - label_len : i
                        + self.input_window
                        + target_window,
                        9,
                    ].tolist(),
                ]
            )

        assert len(self.price) == len(self.y)
        assert len(self.volume) == len(self.date)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "price": torch.tensor(self.price[index]),
            "volume": torch.tensor(self.volume[index]),
            "price_min_max_scaled": torch.tensor(self.price_min_max[index]),
            "volume_min_max_scaled": torch.tensor(self.volume_min_max[index]),
            "price_std": torch.tensor(self.price_std[index]),
            "volume_std": torch.tensor(self.volume_std[index]),
            "date": self.date[index],
            "year": torch.tensor(self.year[index]),
            "month": torch.tensor(self.month[index]),
            "day": torch.tensor(self.day[index]),
            "weekday": torch.tensor(self.weekday[index]),
            "y": torch.tensor(self.y[index]),
            "x_mark": torch.tensor(self.x_mark[index]),
            "y_mark": torch.tensor(self.y_mark[index]),
        }


import numpy as np

def firstzero(lst):
    for idx, value in enumerate(lst):
        if value != 0:
            return idx, value

def lastzero(lst):
    for idx in range(len(lst)-1, -1, -1):
        value = lst[idx]
        if value != 0:
            return idx, value

def linear(lst):
    arr = lst[:]
    arr = list(map(float, arr))
    nzero_lst = []
    ### 0이 아닌 인덱스를 찾기.
    for idx, value in enumerate(arr):
        if value != 0:
            nzero_lst.append(idx)

    ### 0이 아닌 인덱스에 대해 해당하는 리스트의 값을 빼고 칸수만큼 나눠주어서 등차수열을 채운다.
    for j in range(len(nzero_lst) - 1):
        # 연속으로 숫자가 있으면 예외처리
        if nzero_lst[j + 1] - nzero_lst[j] == 1:
            pass
        # 나머지 경우 전부다.
        else:
            diff = nzero_lst[j + 1] - nzero_lst[j]  # 0을 끼고 있는 칸수.
            # diff = arr.index(arr[nzero_lst[j+1]]) - arr.index(arr[nzero_lst[j]]) # 0을 끼고 있는 칸수.
            if diff == 1:
                pass
            else:
                d = (arr[nzero_lst[j + 1]] - arr[nzero_lst[j]]) / diff
                alpha = 1
                for i in range(nzero_lst[j] + 1, nzero_lst[j] + diff):
                    arr[i] = round(arr[nzero_lst[j]] + alpha * d, 1)
                    alpha += 1

    index, value = firstzero(arr)
    naver = np.random.normal(0, value * 0.01, size=index)
    for i, v in enumerate(naver):
        arr[i] = round(value + v, 1)    
    
    index, value = lastzero(arr)
    naver = np.random.normal(0, value * 0.01, size=len(arr) - index - 1)
    for i, v in enumerate(naver):
        arr[i + index + 1] = round(value + v, 1)
    return arr

def make_test_data(csv, target_path, input_window, target_window, st, en):
    st = datetime.strptime(st, "%Y-%m-%d")
    en = datetime.strptime(en, "%Y-%m-%d")
    d = (en - st).days

    for i in range(d + 1):
        base = pd.read_csv(csv)
        today = (st + timedelta(days=i)).strftime("%Y-%m-%d")
        print(today)
        idx = base[(base["date"] == today)].index.values[0]
        base = base.iloc[idx - input_window + 1 : idx + target_window + 1, :]
        assert base.shape[0] == input_window + target_window

        base.to_csv(os.path.join(target_path, "{}.csv".format(today)), index=False)
    print("Done")


def load_data(
    path,
    breed,
    input_window,
    target_window,
    label_len,
    batch_size,
    ratio=0.8,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
):
    """Loads the data"""
    dataset = CustomDataset(path, breed, input_window, target_window, label_len)

    # Split the indices in a stratified way
    indices = np.arange(len(dataset))
    train_indices, valid_indices = train_test_split(indices, train_size=ratio, shuffle=False, random_state=777)
    train_indices = train_indices[:-111] # remove overlapping parts

    # Warp into Subsets and DataLoaders
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=True,
    )

    return (train_loader, valid_loader, dataset)

if __name__ == '__main__':
    """
    For debugging dataset.py
    """
    args = easydict.EasyDict({
        "model" : "Informer",
        "device"    : torch.device("cuda:0"),   
        "input_window" : 112,
        "target_window" : 28,
        "label_len" : 56,
        "target_n"   : 21,
        "learning_rate"  : 1e-4,                   
        "batch_size"    : 32,                   
        "epochs" : 50,               
        "path" : "../data/train.csv",
        "save_path"    : "../models",
        "use_best_model": False,
        "enc_in" : 2, # input feature dim,
        "dec_in" : 1, # output feature dim
        "wandb" : False
    })
    print("Test: load_data...")
    breed= '배추'
    # , '무', '양파', '건고추','마늘',
    #     '대파', '얼갈이배추', '양배추', '깻잎',
    #     '시금치', '미나리', '당근',
    #     '파프리카', '새송이', '팽이버섯', '토마토',
    #     '청상추', '백다다기', '애호박', '캠벨얼리', '샤인마스캇'
    train_loader, valid_loader, scaler = load_data(args.path, breed, args.input_window, args.target_window, args.label_len, batch_size=args.batch_size)
    print(len(train_loader), len(valid_loader))
    train_date = []
    valid_date = []
    for i, data in enumerate(train_loader):
        for batch in data['date']:
            for date in batch:
                train_date.append(datetime.strptime(date, '%Y-%m-%d'))
    print(len(train_date))
    print(max(train_date))
    for i, data in enumerate(valid_loader):
        for batch in data['date']:
            for date in batch:
                valid_date.append(datetime.strptime(date, '%Y-%m-%d'))
    print(len(valid_date))
    print(min(valid_date))
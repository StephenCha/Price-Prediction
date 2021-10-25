import pandas as pd
import numpy as np
from glob import glob
import multiprocessing
import pickle

### AT_TSALET_ALL 파일 전처리 코드
### AT_TSALET_ALL파일로부터 train.csv 파일 생성하는 코드입니다.
### 대상 외 품목/품종 추가, 기타 column 등 추가하여 feature로 활용 가능합니다.

# 윈도우즈 사용시 함수를 별도의 .py파일로 저장 후 import하여 사용
def preprocessing(tsalet_file):
    date = tsalet_file.split("/")[-1].split(".")[0]
    print("Processing ", date)

    unique_pum = [
        "배추",
        "무",
        "양파",
        "건고추",
        "마늘",
        "대파",
        "얼갈이배추",
        "양배추",
        "깻잎",
        "시금치",
        "미나리",
        "당근",
        "파프리카",
        "새송이",
        "팽이버섯",
        "토마토",
    ]
    unique_kind = ["청상추", "백다다기", "애호박", "캠벨얼리", "샤인마스캇"]
    train_dict = {"date": []}
    for sub in unique_pum:
        train_dict[f"{sub}_거래량(kg)"] = []
        train_dict[f"{sub}_가격(원/kg)"] = []

    for sub in unique_kind:
        train_dict[f"{sub}_거래량(kg)"] = []
        train_dict[f"{sub}_가격(원/kg)"] = []

    tsalet_sample = pd.read_csv(tsalet_file)
    try:
        days = sorted(tsalet_sample["SALEDATE"].unique())
    except:
        print("Exception :", date)
        train_dict["date"].append(date)
        for sub in unique_pum:
            train_dict[f"{sub}_거래량(kg)"].append(0)
            train_dict[f"{sub}_가격(원/kg)"].append(0)
        for sub in unique_kind:
            train_dict[f"{sub}_거래량(kg)"].append(0)
            train_dict[f"{sub}_가격(원/kg)"].append(0)
        with open(
            f'../data/private/dict/{tsalet_file.split("/")[-1].split(".")[0]}.pkl', "wb"
        ) as f:
            pickle.dump(train_dict, f)
        return

    for day in days:
        train_dict["date"].append(day)
        for sub in unique_pum:
            # 날짜별, 품목별, 거래량이 0 이상인 행만 선택
            c = tsalet_sample[
                (tsalet_sample["SALEDATE"] == day)
                & (tsalet_sample["PUM_NM"] == sub)
                & (tsalet_sample["TOT_QTY"] > 0)
            ]
            if c.shape[0] == 0:
                train_dict[f"{sub}_거래량(kg)"].append(0)
                train_dict[f"{sub}_가격(원/kg)"].append(0)
            else:
                tot_amt = c["TOT_AMT"].sum().astype(float)
                tot_qty = c["TOT_QTY"].sum().astype(float)
                mean_price = tot_amt / (tot_qty + 1e-20)
                train_dict[f"{sub}_거래량(kg)"].append(tot_qty)
                train_dict[f"{sub}_가격(원/kg)"].append(mean_price)

        for sub in unique_kind:
            # 날짜별, 품종별, 거래량이 0 이상인 행만 선택
            c = tsalet_sample[
                (tsalet_sample["SALEDATE"] == day)
                & (tsalet_sample["KIND_NM"] == sub)
                & (tsalet_sample["TOT_QTY"] > 0)
            ]
            if c.shape[0] == 0:
                train_dict[f"{sub}_거래량(kg)"].append(0)
                train_dict[f"{sub}_가격(원/kg)"].append(0)
            else:
                tot_amt = c["TOT_AMT"].sum().astype(float)
                tot_qty = c["TOT_QTY"].sum().astype(float)
                mean_price = round(tot_amt / (tot_qty + 1e-20))
                tot_qty = round(tot_qty, 1)
                train_dict[f"{sub}_거래량(kg)"].append(tot_qty)
                train_dict[f"{sub}_가격(원/kg)"].append(mean_price)
    with open(
        f'../data/private/dict/{tsalet_file.split("/")[-1].split(".")[0]}.pkl', "wb"
    ) as f:
        pickle.dump(train_dict, f)

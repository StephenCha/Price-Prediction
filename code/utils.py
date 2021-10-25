import random
import os
import torch
import numpy as np
import pandas as pd


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def nmae(answer_df, submission_df):
    """
    평가용 코드
    """
    answer = answer_df.iloc[:, 1:].to_numpy()
    submission = submission_df.iloc[:, 1:].to_numpy()
    target_idx = np.where(answer != 0)

    true = answer[target_idx]
    pred = submission[target_idx]

    score = np.mean(np.abs(true - pred) / true)

    return score


def at_nmae(answer_df, submission_df):
    """
    평가용 코드
    """
    week_1_answer = answer_df.iloc[0::3]
    week_2_answer = answer_df.iloc[1::3]
    week_4_answer = answer_df.iloc[2::3]

    idx_col_nm = answer_df.columns[0]
    week_1_submission = submission_df[
        submission_df[idx_col_nm].isin(week_1_answer[idx_col_nm])
    ]
    week_2_submission = submission_df[
        submission_df[idx_col_nm].isin(week_2_answer[idx_col_nm])
    ]
    week_4_submission = submission_df[
        submission_df[idx_col_nm].isin(week_4_answer[idx_col_nm])
    ]

    score1 = nmae(week_1_answer, week_1_submission)
    score2 = nmae(week_2_answer, week_2_submission)
    score4 = nmae(week_4_answer, week_4_submission)

    score = (score1 + score2 + score4) / 3

    return score

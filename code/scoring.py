import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def nmae(answer_df, submission_df):
    """
    평가용 코드
    """
    answer = answer_df.to_numpy()
    submission = submission_df.to_numpy()
    target_idx = np.where(answer != 0)

    true = answer[target_idx]
    pred = submission[target_idx]

    # print(answer.shape, submission.shape)
    # print(true.shape, pred.shape)

    score = np.mean(np.abs(true - pred) / true)
    return score

def convert_date(date):
    """
    Input:YYYY-MM-DD+Nweek
    Return: YYYY-MM-DD, datetime after N week(s)
    """
    date, week_delta = date.split('+')
    week_delta = int(week_delta[0]) # filtering only number / ex: {N}week -> N
    date = datetime.strptime(date, '%Y-%m-%d') # convert str to datetime
    date = date + timedelta(weeks=week_delta) # get datetime after week_delta
    
    return date.strftime('%Y-%m-%d')

def scoring(answer_df, submission_df):
    """
    평가용 코드
    """
    submission_df.rename(columns={'예측대상일자': 'date'}, inplace=True)
    submission_df.iloc[:,0] = submission_df.iloc[:,0].apply(convert_date)

    # make dataFrame with same date order, but value is equal to answer_df
    submission_df.set_index('date', inplace=True)
    answer_df_submission_order = submission_df.copy()
    answer_df.set_index('date', inplace=True)
    answer_df_submission_order.update(answer_df)

    # split by how many weeks have to pass
    week_1_submission = submission_df.iloc[0::3]
    week_2_submission = submission_df.iloc[1::3]
    week_4_submission = submission_df.iloc[2::3]

    week_1_answer = answer_df_submission_order.iloc[0::3]
    week_2_answer = answer_df_submission_order.iloc[1::3]
    week_4_answer = answer_df_submission_order.iloc[2::3]


    # week_1_answer.to_csv('week_1_answer.csv',encoding='utf-8-sig')
    # week_1_submission.to_csv('week_1_submission.csv',encoding='utf-8-sig')
    # print(week_1_answer.shape, week_1_submission.shape)
    score1 = nmae(week_1_answer, week_1_submission)
    score2 = nmae(week_2_answer, week_2_submission)
    score4 = nmae(week_4_answer, week_4_submission)

    score = (score1 + score2 + score4) / 3

    return score

if __name__ == '__main__':
    ANSWER_PATH = "../data/public_test.csv"
    SUBMISSION_PATH = "../inference/result.csv"
    answer_df = pd.read_csv(ANSWER_PATH)
    submission_df = pd.read_csv(SUBMISSION_PATH)
    print(scoring(answer_df, submission_df))
import pandas as pd
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from PIL import Image

st.set_page_config(layout="wide")

df = pd.read_csv('/Users/hannesreichelt/Desktop/Bundesliga Dataframe/season_22_23.csv', index_col=0)
df2 = pd.read_csv('/Users/hannesreichelt/Desktop/Bundesliga Dataframe/season_21_22.csv', index_col=0)
df3 = pd.read_csv('/Users/hannesreichelt/Desktop/Bundesliga Dataframe/season_20_21.csv', index_col=0)
df4 = pd.read_csv('/Users/hannesreichelt/Desktop/Bundesliga Dataframe/season_19_20.csv', index_col=0)

df.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df4.reset_index(drop=True, inplace=True)

df = pd.concat([df, df2, df3, df4], ignore_index=True)

df = df.drop(['BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH', 'B365AHA', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA'], axis=1)

df = df.drop(['HTHG','HTAG','HTR','HF',	'AF','HC','AC',	'HY','AY','HR',	'AR','HS'	,'AS',	'HST','AST'], axis=1)

df.rename(columns={'FTHG': 'Home Team Goals', 'FTAG': 'Away Team Goals', 'FTR': 'Full Time Result', 'B365H': 'Quote 1',
                   'B365D': 'Quote 0', 'B365A': 'Quote 2'},
          inplace=True)

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  
df.insert(2, 'Weekday', df['Date'].dt.day_name())

df['Date'] = pd.to_datetime(df['Date'], errors='coercefrom datetime import datetime')

df.loc[(df['Date'] >= datetime(2022, 8, 5)) & (df['Date'] <= datetime(2022, 8, 7)), 'Matchday'] = 1
df.loc[(df['Date'] >= datetime(2022, 8, 12)) & (df['Date'] <= datetime(2022, 8, 14)), 'Matchday'] = 2

df.loc[(df['Date'] >= datetime(2022, 8, 19)) & (df['Date'] <= datetime(2022, 8, 21)), 'Matchday'] = 3
df.loc[(df['Date'] >= datetime(2022, 8, 26)) & (df['Date'] <= datetime(2022, 8, 28)), 'Matchday'] = 4
df.loc[(df['Date'] >= datetime(2022, 9, 2)) & (df['Date'] <= datetime(2022, 9, 4)), 'Matchday'] = 5
df.loc[(df['Date'] >= datetime(2022, 9, 9)) & (df['Date'] <= datetime(2022, 9, 11)), 'Matchday'] = 6
df.loc[(df['Date'] >= datetime(2022, 9, 16)) & (df['Date'] <= datetime(2022, 9, 18)), 'Matchday'] = 7
df.loc[(df['Date'] >= datetime(2022, 9, 30)) & (df['Date'] <= datetime(2022, 10, 2)), 'Matchday'] = 8
df.loc[(df['Date'] >= datetime(2022, 10, 7)) & (df['Date'] <= datetime(2022, 10, 9)), 'Matchday'] = 9
df.loc[(df['Date'] >= datetime(2022, 10, 14)) & (df['Date'] <= datetime(2022, 10, 16)), 'Matchday'] = 10
df.loc[(df['Date'] >= datetime(2022, 10, 21)) & (df['Date'] <= datetime(2022, 10, 23)), 'Matchday'] = 11
df.loc[(df['Date'] >= datetime(2022, 10, 28)) & (df['Date'] <= datetime(2022, 10, 30)), 'Matchday'] = 12
df.loc[(df['Date'] >= datetime(2022, 11, 4)) & (df['Date'] <= datetime(2022, 11, 6)), 'Matchday'] = 13
df.loc[(df['Date'] >= datetime(2022, 11, 8)) & (df['Date'] <= datetime(2022, 11, 9)), 'Matchday'] = 14
df.loc[(df['Date'] >= datetime(2022, 11, 11)) & (df['Date'] <= datetime(2022, 11, 13)), 'Matchday'] = 15
df.loc[(df['Date'] >= datetime(2023, 1, 20)) & (df['Date'] <= datetime(2023, 1, 22)), 'Matchday'] = 16
df.loc[(df['Date'] >= datetime(2023, 1, 24)) & (df['Date'] <= datetime(2023, 1, 25)), 'Matchday'] = 17
df.loc[(df['Date'] >= datetime(2023, 1, 27)) & (df['Date'] <= datetime(2023, 2, 28)), 'Matchday'] = 18
df.loc[(df['Date'] >= datetime(2023, 2, 3)) & (df['Date'] <= datetime(2023, 2, 5)), 'Matchday'] = 19
df.loc[(df['Date'] >= datetime(2023, 2, 10)) & (df['Date'] <= datetime(2023, 2, 12)), 'Matchday'] = 20
df.loc[(df['Date'] >= datetime(2023, 2, 17)) & (df['Date'] <= datetime(2023, 2, 19)), 'Matchday'] = 21
df.loc[(df['Date'] >= datetime(2023, 2, 24)) & (df['Date'] <= datetime(2023, 2, 26)), 'Matchday'] = 22
df.loc[(df['Date'] >= datetime(2023, 3, 3)) & (df['Date'] <= datetime(2023, 8, 5)), 'Matchday'] = 23
df.loc[(df['Date'] >= datetime(2023, 3, 10)) & (df['Date'] <= datetime(2023, 3, 12)), 'Matchday'] = 24
df.loc[(df['Date'] >= datetime(2023, 3, 17)) & (df['Date'] <= datetime(2023, 3, 19)), 'Matchday'] = 25
df.loc[(df['Date'] >= datetime(2023, 3, 31)) & (df['Date'] <= datetime(2023, 4, 2)), 'Matchday'] = 26
df.loc[(df['Date'] >= datetime(2023, 4, 8)) & (df['Date'] <= datetime(2023, 4, 9)), 'Matchday'] = 27
df.loc[(df['Date'] >= datetime(2023, 4, 14)) & (df['Date'] <= datetime(2023, 4, 16)), 'Matchday'] = 28
df.loc[(df['Date'] >= datetime(2023, 4, 21)) & (df['Date'] <= datetime(2023, 4, 23)), 'Matchday'] = 29
df.loc[(df['Date'] >= datetime(2023, 4, 28)) & (df['Date'] <= datetime(2023, 4, 30)), 'Matchday'] = 30
df.loc[(df['Date'] >= datetime(2023, 5, 5)) & (df['Date'] <= datetime(2023, 5, 7)), 'Matchday'] = 31
df.loc[(df['Date'] >= datetime(2023, 5, 12)) & (df['Date'] <= datetime(2023, 5, 14)), 'Matchday'] = 32
df.loc[(df['Date'] >= datetime(2023, 5, 19)) & (df['Date'] <= datetime(2023, 5, 21)), 'Matchday'] = 33
df.loc[(df['Date'] >= datetime(2023, 5, 27)) & (df['Date'] <= datetime(2023, 5, 27)), 'Matchday'] = 34

df.loc[(df['Date'] >= datetime(2021, 8, 13)) & (df['Date'] <= datetime(2021, 8, 15)), 'Matchday'] = 1
df.loc[(df['Date'] >= datetime(2021, 8, 20)) & (df['Date'] <= datetime(2021, 8, 22)), 'Matchday'] = 2
df.loc[(df['Date'] >= datetime(2021, 8, 27)) & (df['Date'] <= datetime(2021, 8, 29)), 'Matchday'] = 3
df.loc[(df['Date'] >= datetime(2021, 9, 10)) & (df['Date'] <= datetime(2021, 9, 12)), 'Matchday'] = 4
df.loc[(df['Date'] >= datetime(2021, 9, 17)) & (df['Date'] <= datetime(2021, 9, 19)), 'Matchday'] = 5
df.loc[(df['Date'] >= datetime(2021, 9, 24)) & (df['Date'] <= datetime(2021, 9, 26)), 'Matchday'] = 6
df.loc[(df['Date'] >= datetime(2021, 10, 1)) & (df['Date'] <= datetime(2021, 10, 3)), 'Matchday'] = 7
df.loc[(df['Date'] >= datetime(2021, 10, 15)) & (df['Date'] <= datetime(2021, 10, 17)), 'Matchday'] = 8
df.loc[(df['Date'] >= datetime(2021, 10, 22)) & (df['Date'] <= datetime(2021, 10, 24)), 'Matchday'] = 9
df.loc[(df['Date'] >= datetime(2021, 10, 29)) & (df['Date'] <= datetime(2021, 10, 31)), 'Matchday'] = 10
df.loc[(df['Date'] >= datetime(2021, 11, 5)) & (df['Date'] <= datetime(2021, 11, 7)), 'Matchday'] = 11
df.loc[(df['Date'] >= datetime(2021, 11, 19)) & (df['Date'] <= datetime(2021, 11, 21)), 'Matchday'] = 12
df.loc[(df['Date'] >= datetime(2021, 11, 26)) & (df['Date'] <= datetime(2021, 11, 28)), 'Matchday'] = 13
df.loc[(df['Date'] >= datetime(2021, 12, 3)) & (df['Date'] <= datetime(2021, 12, 5)), 'Matchday'] = 14
df.loc[(df['Date'] >= datetime(2021, 12, 10)) & (df['Date'] <= datetime(2021, 12, 12)), 'Matchday'] = 15
df.loc[(df['Date'] >= datetime(2021, 12, 14)) & (df['Date'] <= datetime(2021, 12, 15)), 'Matchday'] = 16
df.loc[(df['Date'] >= datetime(2021, 12, 17)) & (df['Date'] <= datetime(2021, 12, 19)), 'Matchday'] = 17
df.loc[(df['Date'] >= datetime(2022, 1, 7)) & (df['Date'] <= datetime(2022, 1, 9)), 'Matchday'] = 18
df.loc[(df['Date'] >= datetime(2022, 1, 14)) & (df['Date'] <= datetime(2022, 1, 16)), 'Matchday'] = 19
df.loc[(df['Date'] >= datetime(2022, 1, 21)) & (df['Date'] <= datetime(2022, 1, 23)), 'Matchday'] = 20
df.loc[(df['Date'] >= datetime(2022, 2, 4)) & (df['Date'] <= datetime(2022, 2, 6)), 'Matchday'] = 21
df.loc[(df['Date'] >= datetime(2022, 2, 11)) & (df['Date'] <= datetime(2022, 2, 13)), 'Matchday'] = 22
df.loc[(df['Date'] >= datetime(2022, 2, 18)) & (df['Date'] <= datetime(2022, 2, 20)), 'Matchday'] = 23
df.loc[(df['Date'] >= datetime(2022, 2, 25)) & (df['Date'] <= datetime(2022, 2, 27)), 'Matchday'] = 24

df.loc[
    ((df['Date'] >= datetime(2022, 3, 4)) & (df['Date'] <= datetime(2022, 3, 6))) |
    (df['Date'] == datetime(2022, 3, 16)), 'Matchday'] = 25

df.loc[
    ((df['Date'] >= datetime(2022, 3, 12)) & (df['Date'] <= datetime(2022, 3, 13))) |
    (df['Date'] == datetime(2022, 4, 6)), 'Matchday'] = 26

df.loc[(df['Date'] >= datetime(2022, 3, 18)) & (df['Date'] <= datetime(2022, 3, 20)), 'Matchday'] = 27
df.loc[(df['Date'] >= datetime(2022, 4, 1)) & (df['Date'] <= datetime(2022, 4, 3)), 'Matchday'] = 28
df.loc[(df['Date'] >= datetime(2022, 4, 8)) & (df['Date'] <= datetime(2022, 4,10)), 'Matchday'] = 29
df.loc[(df['Date'] >= datetime(2022, 4, 16)) & (df['Date'] <= datetime(2022, 4, 17)), 'Matchday'] = 30
df.loc[(df['Date'] >= datetime(2022, 4, 22)) & (df['Date'] <= datetime(2022, 4, 24)), 'Matchday'] = 31
df.loc[(df['Date'] >= datetime(2022, 4, 29)) & (df['Date'] <= datetime(2022, 5, 2)), 'Matchday'] = 32
df.loc[(df['Date'] >= datetime(2022, 5, 6)) & (df['Date'] <= datetime(2022, 5, 8)), 'Matchday'] = 33
df.loc[(df['Date'] >= datetime(2022, 5, 14)) & (df['Date'] <= datetime(2022, 5, 14)), 'Matchday'] = 34


df.loc[(df['Date'] >= datetime(2020, 9, 18)) & (df['Date'] <= datetime(2020, 9, 20)), 'Matchday'] = 1
df.loc[(df['Date'] >= datetime(2020, 9, 25)) & (df['Date'] <= datetime(2020, 9, 27)), 'Matchday'] = 2
df.loc[(df['Date'] >= datetime(2020, 10, 2)) & (df['Date'] <= datetime(2020, 10, 4)), 'Matchday'] = 3
df.loc[(df['Date'] >= datetime(2020, 10, 17)) & (df['Date'] <= datetime(2020, 10, 18)), 'Matchday'] = 4
df.loc[(df['Date'] >= datetime(2020, 10, 23)) & (df['Date'] <= datetime(2020, 10, 26)), 'Matchday'] = 5
df.loc[(df['Date'] >= datetime(2020, 10, 30)) & (df['Date'] <= datetime(2020, 11, 2)), 'Matchday'] = 6
df.loc[(df['Date'] >= datetime(2020, 11, 6)) & (df['Date'] <= datetime(2020, 11, 8)), 'Matchday'] = 7
df.loc[(df['Date'] >= datetime(2020, 11, 21)) & (df['Date'] <= datetime(2020, 11, 22)), 'Matchday'] = 8
df.loc[(df['Date'] >= datetime(2020, 11, 27)) & (df['Date'] <= datetime(2020, 11, 29)), 'Matchday'] = 9
df.loc[(df['Date'] >= datetime(2020, 12, 4)) & (df['Date'] <= datetime(2020, 12, 7)), 'Matchday'] = 10
df.loc[(df['Date'] >= datetime(2020, 12, 11)) & (df['Date'] <= datetime(2020, 12, 13)), 'Matchday'] = 11
df.loc[(df['Date'] >= datetime(2020, 12, 15)) & (df['Date'] <= datetime(2020, 12, 16)), 'Matchday'] = 12
df.loc[(df['Date'] >= datetime(2020, 12, 18)) & (df['Date'] <= datetime(2020, 12, 20)), 'Matchday'] = 13
df.loc[(df['Date'] >= datetime(2021, 1, 2)) & (df['Date'] <= datetime(2021, 1, 3)), 'Matchday'] = 14
df.loc[(df['Date'] >= datetime(2021, 1, 8)) & (df['Date'] <= datetime(2021, 1, 10)), 'Matchday'] = 15
df.loc[(df['Date'] >= datetime(2021, 1, 15)) & (df['Date'] <= datetime(2021, 1, 17)), 'Matchday'] = 16
df.loc[(df['Date'] >= datetime(2021, 1, 19)) & (df['Date'] <= datetime(2021, 1, 20)), 'Matchday'] = 17
df.loc[(df['Date'] >= datetime(2021, 1, 22)) & (df['Date'] <= datetime(2021, 1, 24)), 'Matchday'] = 18
df.loc[(df['Date'] >= datetime(2021, 1, 29)) & (df['Date'] <= datetime(2021, 1, 31)), 'Matchday'] = 19
df.loc[(df['Date'] >= datetime(2021, 2, 5)) & (df['Date'] <= datetime(2021, 2, 7)), 'Matchday'] = 20
df.loc[(df['Date'] >= datetime(2021, 3, 10)) & (df['Date'] <= datetime(2021, 3, 10)), 'Matchday'] = 20
df.loc[(df['Date'] >= datetime(2021, 2, 12)) & (df['Date'] <= datetime(2021, 2, 15)), 'Matchday'] = 21
df.loc[(df['Date'] >= datetime(2021, 2, 19)) & (df['Date'] <= datetime(2021, 2, 21)), 'Matchday'] = 22
df.loc[(df['Date'] >= datetime(2021, 2, 26)) & (df['Date'] <= datetime(2021, 2, 28)), 'Matchday'] = 23
df.loc[(df['Date'] >= datetime(2021, 3, 5)) & (df['Date'] <= datetime(2021, 3, 7)), 'Matchday'] = 24
df.loc[(df['Date'] >= datetime(2021, 3, 12)) & (df['Date'] <= datetime(2021, 3, 14)), 'Matchday'] = 25
df.loc[(df['Date'] >= datetime(2021, 3, 19)) & (df['Date'] <= datetime(2021, 3, 21)), 'Matchday'] = 26
df.loc[(df['Date'] >= datetime(2021, 4, 3)) & (df['Date'] <= datetime(2021, 4, 4)), 'Matchday'] = 27
df.loc[(df['Date'] >= datetime(2021, 4, 9)) & (df['Date'] <= datetime(2021, 4, 12)), 'Matchday'] = 28

df.loc[
    ((df['Date'] >= datetime(2021, 4, 16)) & (df['Date'] <= datetime(2021, 4, 18))) |
    (df['Date'] == datetime(2021, 3, 5)), 'Matchday'] = 29

df.loc[
    ((df['Date'] >= datetime(2021, 4, 20)) & (df['Date'] <= datetime(2021, 4, 21))) |
    (df['Date'] == datetime(2021, 5, 6)), 'Matchday'] = 30

df.loc[
    ((df['Date'] >= datetime(2021, 4, 23)) & (df['Date'] <= datetime(2021, 4, 25))) |
    (df['Date'] == datetime(2021, 5, 6)) |
    (df['Date'] == datetime(2021, 5, 3)), 'Matchday'] = 31

df.loc[(df['Date'] >= datetime(2021, 5, 7)) & (df['Date'] <= datetime(2021, 5, 12)), 'Matchday'] = 32
df.loc[(df['Date'] >= datetime(2021, 5, 15)) & (df['Date'] <= datetime(2021, 6, 16)), 'Matchday'] = 33
df.loc[(df['Date'] >= datetime(2021, 5, 22)) & (df['Date'] <= datetime(2021, 5, 22)), 'Matchday'] = 34


df.loc[(df['Date'] >= datetime(2019, 8, 16)) & (df['Date'] <= datetime(2019, 8, 18)), 'Matchday'] = 1
df.loc[(df['Date'] >= datetime(2019, 8, 23)) & (df['Date'] <= datetime(2019, 8, 25)), 'Matchday'] = 2
df.loc[(df['Date'] >= datetime(2019, 8 ,30)) & (df['Date'] <= datetime(2019, 9, 1)), 'Matchday'] = 3
df.loc[(df['Date'] >= datetime(2019, 9, 13)) & (df['Date'] <= datetime(2019, 9, 15)), 'Matchday'] = 4
df.loc[(df['Date'] >= datetime(2019, 9, 20)) & (df['Date'] <= datetime(2019, 9, 23)), 'Matchday'] = 5
df.loc[(df['Date'] >= datetime(2019, 9, 27)) & (df['Date'] <= datetime(2019, 9, 29)), 'Matchday'] = 6
df.loc[(df['Date'] >= datetime(2019, 10, 4)) & (df['Date'] <= datetime(2019, 10, 6)), 'Matchday'] = 7
df.loc[(df['Date'] >= datetime(2019, 10, 18)) & (df['Date'] <= datetime(2019, 10, 20)), 'Matchday'] = 8
df.loc[(df['Date'] >= datetime(2019, 10, 25)) & (df['Date'] <= datetime(2019, 10, 27)), 'Matchday'] = 9
df.loc[(df['Date'] >= datetime(2019, 11, 1)) & (df['Date'] <= datetime(2019, 11, 3)), 'Matchday'] = 10
df.loc[(df['Date'] >= datetime(2019, 11, 8)) & (df['Date'] <= datetime(2019, 11, 10)), 'Matchday'] = 11
df.loc[(df['Date'] >= datetime(2019, 11, 22)) & (df['Date'] <= datetime(2019, 11, 24)), 'Matchday'] = 12
df.loc[(df['Date'] >= datetime(2019, 11, 29)) & (df['Date'] <= datetime(2019, 12, 2)), 'Matchday'] = 13
df.loc[(df['Date'] >= datetime(2019, 12, 6)) & (df['Date'] <= datetime(2019, 12, 8)), 'Matchday'] = 14
df.loc[(df['Date'] >= datetime(2019, 12, 13)) & (df['Date'] <= datetime(2019, 12, 15)), 'Matchday'] = 15
df.loc[(df['Date'] >= datetime(2019, 12, 17)) & (df['Date'] <= datetime(2019, 12, 18)), 'Matchday'] = 16
df.loc[(df['Date'] >= datetime(2019, 12, 20)) & (df['Date'] <= datetime(2019, 12, 22)), 'Matchday'] = 17
df.loc[(df['Date'] >= datetime(2020, 1, 17)) & (df['Date'] <= datetime(2020, 1, 19)), 'Matchday'] = 18
df.loc[(df['Date'] >= datetime(2020, 1, 24)) & (df['Date'] <= datetime(2020, 1, 26)), 'Matchday'] = 19
df.loc[(df['Date'] >= datetime(2020, 1, 31)) & (df['Date'] <= datetime(2020, 2, 2)), 'Matchday'] = 20

df.loc[
    ((df['Date'] >= datetime(2020, 2, 7)) & (df['Date'] <= datetime(2020, 2, 9))) |
    (df['Date'] == datetime(2020, 3, 11)),
    'Matchday'
] = 21

df.loc[(df['Date'] >= datetime(2020, 2, 14)) & (df['Date'] <= datetime(2020, 2, 16)), 'Matchday'] = 22
df.loc[(df['Date'] >= datetime(2020, 2, 21)) & (df['Date'] <= datetime(2020, 2, 24)), 'Matchday'] = 23

df.loc[
    ((df['Date'] >= datetime(2020, 2, 28)) & (df['Date'] <= datetime(2020, 3, 1))) |
    (df['Date'] == datetime(2020, 6, 3)),
    'Matchday'
] = 24

df.loc[
    ((df['Date'] >= datetime(2020, 3, 6)) & (df['Date'] <= datetime(2020, 3, 8))) |
    ((df['Date'] >= datetime(2020, 3, 16)) & (df['Date'] <= datetime(2020, 3, 18))),
    'Matchday'
] = 25

df.loc[(df['Date'] >= datetime(2020, 5, 16)) & (df['Date'] <= datetime(2020, 5, 18)), 'Matchday'] = 26
df.loc[(df['Date'] >= datetime(2020, 5, 22)) & (df['Date'] <= datetime(2020, 5, 24)), 'Matchday'] = 27
df.loc[(df['Date'] >= datetime(2020, 5, 26)) & (df['Date'] <= datetime(2020, 5, 27)), 'Matchday'] = 28
df.loc[(df['Date'] >= datetime(2020, 5, 29)) & (df['Date'] <= datetime(2020, 6, 1)), 'Matchday'] = 29
df.loc[(df['Date'] >= datetime(2020, 6, 5)) & (df['Date'] <= datetime(2020, 6, 7)), 'Matchday'] = 30
df.loc[(df['Date'] >= datetime(2020, 6, 12)) & (df['Date'] <= datetime(2020, 6, 14)), 'Matchday'] = 31
df.loc[(df['Date'] >= datetime(2020, 6, 16)) & (df['Date'] <= datetime(2020, 6, 17)), 'Matchday'] = 32
df.loc[(df['Date'] >= datetime(2020, 6, 20)) & (df['Date'] <= datetime(2020, 6, 20)), 'Matchday'] = 33
df.loc[(df['Date'] >= datetime(2020, 6, 27)) & (df['Date'] <= datetime(2020, 6, 27)), 'Matchday'] = 34

df['Matchday'] = df['Matchday'].astype(int)
cols = df.columns.tolist()
cols = ['Matchday'] + cols[:-1]
df = df.reindex(columns=cols)

def num_results(df):
    df['Full Time Result'] = df['Full Time Result'].map({'H': 1, 'D': 0, 'A': 2})
    return df

df = df.loc[:, ~df.columns.duplicated()]

df = num_results(df)

def add_return(df):


    for index, row in df.iterrows():
        if row['Full Time Result'] == 1:
            df.at[index, 'Return'] = row['Quote 1']
        elif row['Full Time Result'] == 0:
            df.at[index, 'Return'] = row['Quote 0']
        elif row['Full Time Result'] == 2:
            df.at[index, 'Return'] = row['Quote 2']

    return df

add_return(df)

df = add_return(df)

   
def over_under(df):
    df.rename(columns={'B365>2.5': 'Quote >2.5', 'B365<2.5': 'Quote <2.5'}, inplace=True)
    df['Total Goals'] = df['Home Team Goals'] + df['Away Team Goals']
    df['Over/Under Return'] = 0.0
    df.loc[df['Total Goals'] > 2.5, 'Over/Under Return'] = df['Quote >2.5']
    df.loc[df['Total Goals'] < 2.5, 'Over/Under Return'] = df['Quote <2.5']
    df['Over/Under Result'] = 0
    df.loc[df['Total Goals'] > 2.5, 'Over/Under Result'] = 1
    return df

over_under(df)

df = over_under(df)

def get_season(row):
    if pd.Timestamp('2022-08-05') <= row['Date'] <= pd.Timestamp('2023-05-27'):
        return 1
    elif pd.Timestamp('2021-08-13') <= row['Date'] <= pd.Timestamp('2022, 5, 14'):
        return 2
    elif pd.Timestamp('2020-09-18') <= row['Date'] <= pd.Timestamp('2021-05-22'):
        return 3
    elif pd.Timestamp('2019-08-16') <= row['Date'] <= pd.Timestamp('2020-06-27'):
        return 4
    else:
        return None

# Apply the function to create the 'Season' column
df['Season'] = df.apply(get_season, axis=1).astype('Int64')



def team_data_func(df):
    team_names = ['Bayern Munich']
 

    team_data_frames = {}

    for team_name in team_names:
        team_df = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
        team_df.loc[team_df['HomeTeam'] == team_name, 'Opponent'] = team_df['AwayTeam']
        team_df.loc[team_df['AwayTeam'] == team_name, 'Opponent'] = team_df['HomeTeam']
        
        team_data_frames[team_name] = team_df

    return team_data_frames

team_data_frames_for_df = team_data_func(df)

import pandas as pd

def team_data_func2(df):
    team_names = ['Ein Frankfurt', 'Bayern Munich', 'Hertha', 'Augsburg', 'Freiburg', 'Bochum', 'Mainz',
                  "M'gladbach", 'Hoffenheim', 'Union Berlin', 'Wolfsburg', 'Werder Bremen', 'Dortmund',
                  'Leverkusen', 'Stuttgart', 'RB Leipzig', 'FC Koln', 'Schalke 04']

    team_data_frames = {}  # Dictionary to hold the data for each team

    for team_name in team_names:
        # Filter the data for the current team (home and away matches)
        team_df = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()

        # Add the 'Opponent' column to represent the opposing team for each match
        team_df.loc[team_df['HomeTeam'] == team_name, 'Opponent'] = team_df['AwayTeam']
        team_df.loc[team_df['AwayTeam'] == team_name, 'Opponent'] = team_df['HomeTeam']

        # Add other team-specific features
        team_df['Venue'] = team_df.apply(lambda row: 'Home' if row['HomeTeam'] == team_name else 'Away', axis=1)
        team_df['Team Goals Scored'] = team_df.apply(lambda row: row['Home Team Goals'] if row['HomeTeam'] == team_name else row['Away Team Goals'], axis=1)
        team_df['Team Goals Received'] = team_df.apply(lambda row: row['Away Team Goals'] if row['HomeTeam'] == team_name else row['Home Team Goals'], axis=1)
        team_df['venue_codes'] = team_df['Venue'].astype('category').cat.codes
        team_df['opponent_codes'] = team_df['Opponent'].astype('category').cat.codes
        team_df['hour_codes'] = team_df['Time'].str.replace(':.+', '', regex=True).astype('int')
        team_df['week_codes'] = team_df['Date'].dt.dayofweek
        team_df['target'] = (team_df['Full Time Result'] == 0).astype('int')

        # Calculate team points and map them to the result (W/D/L)
        team_df['Team Points'] = team_df.apply(lambda row: 3 if (row['HomeTeam'] == team_name and row['Full Time Result'] == 1) else 0 if (row['HomeTeam'] == team_name and row['Full Time Result'] == 2) else 3 if (row['AwayTeam'] == team_name and row['Full Time Result'] == 2) else 0 if (row['AwayTeam'] == team_name and row['Full Time Result'] == 1) else 1, axis=1)
        result_mapping = {3: 'W', 1: 'D', 0: 'L'}
        team_df['Team Result'] = team_df['Team Points'].map(result_mapping)

        # Apply the function to create the 'Season' column for each match
        team_df['Season'] = team_df.apply(get_season, axis=1).astype('Int64')

        # Store the DataFrame for the current team in the dictionary
        team_data_frames[team_name] = team_df

    return team_data_frames

# Assuming you have 'combined_df' as your DataFrame
team_data_frames_for_df = team_data_func2(df)


def calculate_non_draw_lengths(team_df):
    # Get the 'Team Result' column as a list
    results = team_df['Team Result'].tolist()

    # Initialize variables to track non-draw lengths
    non_draw_lengths = []
    current_length = 0
    # Iterate through the results list
    for result in results:
        if result != 'D':
            current_length += 1
        elif current_length > 0:
            non_draw_lengths.append(current_length)
            current_length = 0
    non_draw_lengths.reverse()
    return non_draw_lengths

def calculate_non_draw_lengths_for_all_teams(team_data_frames):
    non_draw_lengths_per_team = {}

    for team_name, team_df in team_data_frames.items():
        non_draw_lengths = calculate_non_draw_lengths(team_df)
        non_draw_lengths_per_team[team_name] = non_draw_lengths

    return non_draw_lengths_per_team


def calculate_non_draw_lengths_greater_than_one(team_df):
    # Get the 'Team Result' column as a list
    results = team_df['Team Result'].tolist()

    # Initialize variables to track non-draw lengths
    non_draw_lengths = []
    current_length = 0

    # Iterate through the results list and count non-draw streaks bigger than 1
    for result in results:
        if result != 'D':
            current_length += 1
        elif current_length > 1:  # Filter streaks bigger than 1
            non_draw_lengths.append(current_length)
            current_length = 0

    if current_length > 1:
        non_draw_lengths.append(current_length)

    non_draw_lengths.reverse()
    return non_draw_lengths

# Assuming you already have the team_data_frames_for_df dictionary containing data for all teams
non_draw_lengths_for_teams = calculate_non_draw_lengths_for_all_teams(team_data_frames_for_df)

# Assuming you already have the team_data_frames_for_df dictionary containing data for all teams
non_draw_lengths_for_teams = calculate_non_draw_lengths_for_all_teams(team_data_frames_for_df)

def calculate_mean_and_max_non_draw_lengths(team_df, season, n):
    team_df_selected_season = team_df[team_df['Season'] == season]

    if not team_df_selected_season.empty:
        non_draw_lengths = calculate_non_draw_lengths(team_df_selected_season)

        if len(non_draw_lengths) > 0:
            mean_length = int(sum(non_draw_lengths) / len(non_draw_lengths))
            max_length = max(non_draw_lengths)
            return mean_length, max_length
        else:
            return 0, 0  # No non-draw streaks for the selected team and season
    else:
        return -1, -1  # No data available for the selected team and season
def create_n_non_draw_target(team_df, n):
    non_draw_lengths = calculate_non_draw_lengths(team_df)
    n_non_draw_target = [1 if streak >= n else 0 for streak in non_draw_lengths]
    return n_non_draw_target

def betting_strategy(team_df, initial_stake, max_bets, dont_start_func):

    stake = initial_stake
    investment = initial_stake
    cashout = 0
    bet_count = 0
    streak_duration = 0
    streak_start_matchday = None
    total_investment = initial_stake 
    total_cashout = 0

    for index, row in team_df.iterrows():
        if row['Matchday'] > dont_start_func and streak_duration <= 0:
            st.markdown(f"<span style='font-size: 16px;color: orange;'>No new bets were placed after Matchday {dont_start_func} until the end of the season</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Final Summary</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Investment: {total_investment:,.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Return: {total_cashout:,.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Profit: {total_cashout - total_investment:,.2f}", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>ROI: {int(((total_cashout - total_investment) / total_investment) * 100)}%", unsafe_allow_html=True)
            if total_cashout > total_investment:  
                return True
            else:
                return False

        if row['Team Result'] == 'D':
            st.markdown(f"<span style='font-size: 16px; font-weight: bold; color: green;'>Won! Final stake: {stake:,.2f}</span>", unsafe_allow_html=True)
            cashout += stake * row['Quote 0']  
            streak_duration = bet_count + 1
            st.write(f"Investment: {investment:,.2f}")
            st.write(f"Return: {cashout:,.2f}")
            st.write(f"Streak Duration: {streak_duration}")
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Profit: {cashout - investment:,.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>ROI: {int(((cashout - investment) / investment) * 100)}%", unsafe_allow_html=True)

            total_cashout += cashout
            stake = initial_stake
            investment = initial_stake
            cashout = 0
            bet_count = 0
            streak_duration = 0

        else:
            if bet_count >= max_bets:
                st.markdown(f"<span style='font-size: 18px;font-weight: bold;color: Red;'>Reached maximum bets or end of season</span>", unsafe_allow_html=True)
                st.write(f"Investment: {investment:,.2f}")
                st.write(f"Return: {cashout:,.2f}")
                st.write(f"Streak Duration: {streak_duration:,.2f}")
                st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Loss: {investment - cashout:,.2f}</span>", unsafe_allow_html=True)
                if total_cashout > total_investment:  
                    return True
                else:
                    return False

            if streak_start_matchday is None:
                streak_start_matchday = row['Matchday']

            stake *= 2
            investment += stake
            total_investment += stake
            bet_count += 1
            streak_duration += 1
            st.markdown(f"<span style='font-size: 14px; font-weight: bold; color: grey;'>Bet {bet_count}: Lost. Doubling stake to {stake:,.2f}</span>", unsafe_allow_html=True)

    total_cashout += cashout
    st.markdown(f"<span style='font-size: 18px;font-weight: bold;'>Reached maximum bets or end of season</span>", unsafe_allow_html=True)
    st.write(f"Investment: {investment:,.2f}")
    st.write(f"Return: {cashout:,.2f}")
    st.write(f"Streak Duration: {streak_duration:,.2f}")
    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Loss: {investment - cashout:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 18px; font-weight: bold;'>Final Summary</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Investment: {total_investment:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Return: {total_cashout:,.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Total Profit: {total_cashout - total_investment:,.2f}", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>ROI: {int(((total_cashout - total_investment) / total_investment) * 100)}%", unsafe_allow_html=True)
    if total_cashout > total_investment:  
        return True
    else:
        return False
   


   

def plot_season_results_bar(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        wins = team_df_selected_season['Team Result'].eq('W').sum()
        losses = team_df_selected_season['Team Result'].eq('L').sum()
        draws = team_df_selected_season['Team Result'].eq('D').sum()

        x_values = ['Wins', 'Losses', 'Draws']
        y_values = [wins, losses, draws]

        colors = ['green', 'red', 'grey']

        fig = go.Figure(go.Bar(x=x_values, y=y_values, marker=dict(color=colors)))
        fig.update_layout(
            xaxis_title='Result',
            yaxis_title='Number of Matches',
            title=f'Season Results for {team_name} - Season {selected_season}',
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name} in Season {selected_season}")




def plot_season_results_stacked(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        season_results = team_df_selected_season.groupby(['Matchday', 'Team Result'])['Team Result'].count().unstack(fill_value=0)
        color_order = ['D', 'L', 'W']
        season_results = season_results.reindex(columns=color_order, fill_value=0)

        opponents_and_venues = []
        for _, row in season_results.iterrows():
            matchday = row.name
            home_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'HomeTeam'].values[0]
            away_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'AwayTeam'].values[0]
            opponent = home_team if team_name == away_team else away_team
            venue = "H" if team_name == home_team else "A"
            opponents_and_venues.append(f"{opponent} ({venue})")

        fig = px.bar(season_results, x=opponents_and_venues, y=color_order,
                     labels={'x': 'Opponent and Venue', 'y': 'Number of Matches'},
                     title=f'Season Results for {team_name} - Opponent and Venue (Season {selected_season})',
                     color_discrete_map={'D': 'grey', 'L': 'red', 'W': 'green'},
                     barmode='stack')

        fig.update_yaxes(ticks='', showticklabels=False)
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name} in Season {selected_season}")

def plot_over_under_return_by_matchday_plotly(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        over_under_return_by_matchday = team_df_selected_season.groupby('Matchday')['Over/Under Return'].mean().reset_index()
        quote_over_2_5_by_matchday = team_df_selected_season.groupby('Matchday')['Quote >2.5'].mean().reset_index()

        opponents_and_venues = []
        for _, row in over_under_return_by_matchday.iterrows():
            matchday = row['Matchday']
            home_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'HomeTeam'].values[0]
            away_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'AwayTeam'].values[0]
            opponent = home_team if team_name == away_team else away_team
            venue = "H" if team_name == home_team else "A"
            opponents_and_venues.append(f"{opponent} ({venue})")

        bar_color = ['green' if o_u_return == quote_over_2_5 else 'red'
                     for o_u_return, quote_over_2_5 in zip(over_under_return_by_matchday['Over/Under Return'], quote_over_2_5_by_matchday['Quote >2.5'])]

        fig = go.Figure(go.Bar(x=opponents_and_venues, y=[1]*len(over_under_return_by_matchday),
                               marker_color=bar_color))
        fig.update_layout(
            title=f'Over(green)/Under(red) by {team_name} in Season {selected_season})',
            xaxis_title='',
            showlegend=False,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_gridcolor='rgba(0, 0, 0, 0.1)',
            xaxis_tickangle=45,
            legend_title_text='Over/Under Return',
            yaxis=dict(ticks='', showticklabels=False)
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name}")

import plotly.graph_objects as go

def plot_goals_scored_by_matchday_plotly(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        goals_scored_by_matchday = team_df_selected_season.groupby('Matchday')['Team Goals Scored'].sum().reset_index()

        opponents_and_venues = []
        for _, row in goals_scored_by_matchday.iterrows():
            matchday = row['Matchday']
            home_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'HomeTeam'].values[0]
            away_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'AwayTeam'].values[0]
            opponent = home_team if team_name == away_team else away_team
            venue = "H" if team_name == home_team else "A"
            opponents_and_venues.append(f"{opponent} ({venue})")

        goals_scored_by_matchday['Opponent and Venue'] = opponents_and_venues

        fig = go.Figure(go.Bar(x=goals_scored_by_matchday['Opponent and Venue'], y=goals_scored_by_matchday['Team Goals Scored'], marker=dict(color='green')))
        fig.update_layout(
            title=f'Goals Scored by {team_name} - (Season {selected_season})',
            xaxis_title='',
            yaxis_title='Goals Scored',
            showlegend=False,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_gridcolor='rgba(0, 0, 0, 0.1)'
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name}")


def plot_goals_received_by_matchday_plotly(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        goals_received_by_matchday = team_df_selected_season.groupby('Matchday')['Team Goals Received'].sum().reset_index()

        opponents_and_venues = []
        for _, row in goals_received_by_matchday.iterrows():
            matchday = row['Matchday']
            home_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'HomeTeam'].values[0]
            away_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'AwayTeam'].values[0]
            opponent = home_team if team_name == away_team else away_team
            venue = "H" if team_name == home_team else "A"
            opponents_and_venues.append(f"{opponent} ({venue})")

        goals_received_by_matchday['Opponent and Venue'] = opponents_and_venues

        fig = go.Figure(go.Bar(x=goals_received_by_matchday['Opponent and Venue'], y=goals_received_by_matchday['Team Goals Received'], marker=dict(color='red')))
        fig.update_layout(
            title=f'Goals Received by {team_name} - (Season {selected_season})',
            xaxis_title='',
            yaxis_title='Goals Received',
            showlegend=False,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_gridcolor='rgba(0, 0, 0, 0.1)',
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name}")


import plotly.graph_objects as go

def plot_total_goals_by_matchday_plotly(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:
        total_goals_by_matchday = team_df_selected_season.groupby('Matchday')[['Home Team Goals', 'Away Team Goals']].sum().reset_index()
        total_goals_by_matchday['Total Goals'] = total_goals_by_matchday['Home Team Goals'] + total_goals_by_matchday['Away Team Goals']

        opponents_and_venues = []
        for _, row in total_goals_by_matchday.iterrows():
            matchday = row['Matchday']
            home_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'HomeTeam'].values[0]
            away_team = team_df_selected_season.loc[team_df_selected_season['Matchday'] == matchday, 'AwayTeam'].values[0]
            opponent = home_team if team_name == away_team else away_team
            venue = "H" if team_name == home_team else "A"
            opponents_and_venues.append(f"{opponent} ({venue})")

        total_goals_by_matchday['Opponent and Venue'] = opponents_and_venues

        # Define a list of colors based on the y-values (total goals)
        colors = ['grey' if goals > 3 else 'lightgrey' for goals in total_goals_by_matchday['Total Goals']]

        fig = go.Figure(go.Bar(x=total_goals_by_matchday['Opponent and Venue'], 
                               y=total_goals_by_matchday['Total Goals'], 
                               marker=dict(color=colors)))
        fig.update_layout(
            title=f'Total Goals by {team_name} - (Season {selected_season})',
            xaxis_title='',
            yaxis_title='Total Goals',
            showlegend=False,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            yaxis_gridcolor='rgba(0, 0, 0, 0.1)',
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {team_name}")


    

def plot_non_draw_streaks_for_teams(non_draw_lengths_for_teams, team_names):
    if not team_df_selected_season.empty:
        for team_name in team_names:
            non_draw_streaks = non_draw_lengths_for_teams.get(team_name, [])
            match_indices = list(range(1, len(non_draw_streaks) + 1))

            fig = go.Figure(go.Bar(x=match_indices, y=non_draw_streaks, marker_color='grey'))

            # Customize the plot layout
            fig.update_layout(
                title=f"All Season Streaks for {team_name} starting from 19/20",
                xaxis_title="",
                yaxis_title="Non-Draw Streak Length",
                showlegend=False,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                yaxis_gridcolor='rgba(0, 0, 0, 0.1)',
            )

            fig.update_xaxes(showticklabels=False)
            # Show the plot for each team separately
            st.plotly_chart(fig)
    else:
        st.write(f"No data available for team: {selected_team}")


def calculate_mean_goals_scored(team_df_selected_season, team_name):
    if not team_df_selected_season.empty:
        mean_goals_scored = np.mean(team_df_selected_season['Team Goals Scored'])
        return mean_goals_scored
    else:
        return None

def calculate_mean_goals_received(team_df_selected_season, team_name):
    if not team_df_selected_season.empty:
        mean_goals_received = np.mean(team_df_selected_season['Team Goals Received'])
        return mean_goals_received
    else:
        return None

def calculate_mean_total_goals(team_df_selected_season, team_name):
    if not team_df_selected_season.empty:
        team_df_selected_season['Total Goals'] = team_df_selected_season['Home Team Goals'] + team_df_selected_season['Away Team Goals']
        mean_total_goals = np.mean(team_df_selected_season['Total Goals'])
        return mean_total_goals
    else:
        return None

        
def display_team_data(team_df_selected_season, team_name, columns_to_display):
    if not team_df_selected_season.empty:
        columns_to_display = [col for col in columns_to_display if col in team_df_selected_season.columns]

        if columns_to_display:
            st.dataframe(team_df_selected_season[columns_to_display])
        else:
            st.write("No matching columns to display.")
    else:
        st.write(f"No data available for team: {team_name}")

def plot_over_under_return_vs_target_heatmap(team_df_selected_season, team_name, selected_season):
    if not team_df_selected_season.empty:

        # Assuming the columns are named "target" and "Over/Under Result"
        contingency_table = pd.crosstab(team_df_selected_season["target"], team_df_selected_season["Over/Under Result"])
        correlation = contingency_table.iloc[1, 1] / contingency_table.sum(axis=0)[1]

        colorscale = [[0, 'lightgreen'], [1, 'green']]

        # Create the heatmap figure using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=contingency_table.values,
            x=['<2.5', '>2.5'],
            y=['Non-Draw', 'Draw'],
            colorscale=colorscale,
            showscale=False
        ))

        for i in range(len(contingency_table)):
            for j in range(len(contingency_table.columns)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(contingency_table.values[i][j]),
                    showarrow=False,
                    font=dict(color='black', size=14)
                )

       
        fig.update_layout(
            title=f"Correlation - Over/Under Result vs.Draw ({team_name})",
            xaxis_title="Over/Under Result",
            yaxis_title="target",
            xaxis=dict(tickvals=[0, 1], ticktext=['<2.5', '>2.5']),
            yaxis=dict(tickvals=[0, 1], ticktext=['Non-Draw', 'Draw']),
            width=400,
            height=400
        )
        st.plotly_chart(fig)
    else:
        st.write(f"No data available for {team_name} in Season {selected_season} or 'Over/Under Result' column is missing.")




team_names = ['Ein Frankfurt', 'Bayern Munich', 'Hertha', 'Augsburg', 'Freiburg', 'Bochum', 'Mainz',
              "M'gladbach", 'Hoffenheim', 'Union Berlin', 'Wolfsburg', 'Werder Bremen', 'Dortmund',
              'Leverkusen', 'Stuttgart', 'RB Leipzig', 'FC Koln', 'Schalke 04']

team_mapping = {
    "Heidenheim": team_data_frames_for_df["Hertha"],
    "Darmstadt": team_data_frames_for_df["Schalke 04"],
}


def train_rf_model(team_names, non_draw_lengths_for_teams, team_data_frames_for_df, n):

    team_mapping = {
    "Heidenheim": team_data_frames_for_df["Hertha"],
    "Darmstadt": team_data_frames_for_df["Schalke 04"],
    }       
    team_accuracies = {}
    team_predictions = {}
    team_probabilities = {}

    for team_name in team_names:
        non_draw_lengths = non_draw_lengths_for_teams.get(team_name)
        if non_draw_lengths is not None and len(non_draw_lengths) > 0:
            team_df = team_data_frames_for_df.get(team_name)
            target = create_n_non_draw_target(team_df, n)
            X = np.array(non_draw_lengths).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=42)
            n_estimators = 10
            random_state = 42
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            team_accuracies[team_name] = accuracy
            team_predictions[team_name] = y_pred
            y_prob = rf_model.predict_proba(X_test)
            num_classes = y_prob.shape[1]
            if num_classes == 1:
                draw_streak_probabilities = 1 - y_prob[:, 0]
            else:
                draw_streak_probabilities = 1 - y_prob[:, 0]
            team_probabilities[team_name] = draw_streak_probabilities

    team_average_probabilities = {}
    for team_name, probabilities in team_probabilities.items():
        average_probability = np.mean(probabilities) * 100
        team_average_probabilities[team_name] = average_probability

    mean_accuracy = np.mean(list(team_accuracies.values()))
    return team_average_probabilities, mean_accuracy

import streamlit as st
def team_images(selected_team):
    if selected_team == 'Werder Bremen':
        werder_logo = '/Users/hannesreichelt/Desktop/1.png'
        st.image(werder_logo, width=200)

    elif selected_team == 'Ein Frankfurt':
        frankfurt_logo = '/Users/hannesreichelt/Desktop/2.png'
        st.image(frankfurt_logo, width=200)

    elif selected_team == 'Bayern Munich':
        bayern_logo = '/Users/hannesreichelt/Desktop/5.png'
        st.image(bayern_logo, width=200)

    elif selected_team == 'Hertha':
        hertha_logo = '/Users/hannesreichelt/Desktop/8.png'
        st.image(hertha_logo, width=200)

    elif selected_team == 'Augsburg':
        augsburg_logo = '/Users/hannesreichelt/Desktop/9.png'
        st.image(augsburg_logo, width=200)

    elif selected_team == 'Freiburg':
        freiburg_logo = '/Users/hannesreichelt/Desktop/10.png'
        st.image(freiburg_logo, width=200)

    elif selected_team == 'Bochum':
        bochum_logo = '/Users/hannesreichelt/Desktop/15.png'
        st.image(bochum_logo, width=200)

    elif selected_team == 'Mainz':
        mainz_logo = '/Users/hannesreichelt/Desktop/11.png'
        st.image(mainz_logo, width=200)

    elif selected_team == "M'gladbach":
        gladbach_logo = '/Users/hannesreichelt/Desktop/18.png'
        st.image(gladbach_logo, width=200)

    elif selected_team == 'Hoffenheim':
        hoffenheim_logo = '/Users/hannesreichelt/Desktop/20.png'
        st.image(hoffenheim_logo, width=200)

    elif selected_team == 'Union Berlin':
        union_logo = '/Users/hannesreichelt/Desktop/19.png'
        st.image(union_logo, width=200)

    elif selected_team == 'Wolfsburg':
        wolfsburg_logo = '/Users/hannesreichelt/Desktop/13.png'
        st.image(wolfsburg_logo, width=200)

    elif selected_team == 'Dortmund':
        dortmund_logo = '/Users/hannesreichelt/Desktop/7.png'
        st.image(dortmund_logo, width=200)

    elif selected_team == 'Leverkusen':
        leverkusen_logo = '/Users/hannesreichelt/Desktop/6.png'
        st.image(leverkusen_logo, width=200)

    elif selected_team == 'Stuttgart':
        stuttgart_logo = '/Users/hannesreichelt/Desktop/4.png'
        st.image(stuttgart_logo, width=200)

    elif selected_team == 'RB Leipzig':
        leipzig_logo = '/Users/hannesreichelt/Desktop/RB_Leipzig_2014_logo.svg.png'
        st.image(leipzig_logo, width=200)

    elif selected_team == 'FC Koln':
        koln_logo = '/Users/hannesreichelt/Desktop/3.png'
        st.image(koln_logo, width=200)

    elif selected_team == 'Schalke 04':
        schalke_logo = '/Users/hannesreichelt/Desktop/12.png'
        st.image(schalke_logo, width=200)

    elif selected_team == 'Heidenheim':
        heidenheim_logo = '/Users/hannesreichelt/Desktop/16.png'
        st.image(heidenheim_logo, width=200)

    elif selected_team == 'Darmstadt':
        darmstadt_logo = '/Users/hannesreichelt/Desktop/SV_Darmstadt_98_Logo.svg.png'
        st.image(darmstadt_logo, width=200)





   




Hero_Image = "/Users/hannesreichelt/Desktop/hero.png"
st.image(Hero_Image)
st.write("")
st.write("")
st.write("This is a simple and straight forward betting system for the German Bundesliga.Just choose a team and bet on DRAW. If you loose the bet just repeat with a double stake.It is pretty self explanatory and easy to understand. However please use this just for fun or research. Gambling sucks.")
st.write("")
st.write("Please choose a team and season to see the statistical data of the last four seasons:")


team_names = ['Ein Frankfurt', 'Bayern Munich', 'Hertha', 'Augsburg', 'Freiburg', 'Bochum', 'Mainz',
              "M'gladbach", 'Hoffenheim', 'Union Berlin', 'Wolfsburg', 'Werder Bremen', 'Dortmund',
              'Leverkusen', 'Stuttgart', 'RB Leipzig', 'FC Koln', 'Schalke 04',]

selected_team = st.selectbox('Select a team:', team_names, key='team_selectbox')


seasons = [1, 2, 3, 4]

selected_season = st.selectbox('Select a season:', seasons, key='season_selectbox')
team_df = team_data_frames_for_df.get(selected_team)

if team_df is not None:
    if selected_season in seasons:
        team_df_selected_season = team_df[team_df['Season'] == selected_season]

        if not team_df_selected_season.empty:
            col1, col42,col2  = st.columns([2,0.7,1.3])

            with col1:
                plot_season_results_bar(team_df_selected_season, selected_team, selected_season)
            
            with col42:
                st.write("")

            with col2:
                st.write("")
                with st.container(): 
                    team_images(selected_team)
                st.write("")

                mean_length, max_length = calculate_mean_and_max_non_draw_lengths(team_df, selected_season, n=9)
                if mean_length == -1:
                    st.write(f"No data available for {selected_team} in Season {selected_season}.")
                elif mean_length == 0:
                    st.write(f"No non-draw streaks for {selected_team} in Season {selected_season}.")
                else:
                    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Mean non-draw streak length in 4 Seasons : {mean_length}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Max non-draw streak length in 4 Seasons : {max_length}</span>", unsafe_allow_html=True)
                    
 

            st.write("First we need to select a initial stake. This stake will be double after each loss. In a casino this would call 'Martingale System'. However in Football the odds on draw a usually 3x instead of 2x. Therefore the risk is higher.")
            st.write("")
            st.write("Second we no determine or maximun lenght of best. This is basically the question of how much risk we are willing to take. A long streak can offer huge returns but also requires a certain investmens. It crucial to keep this in mind. At a certain point the stake will raise exponentially.")
            st.write("")
            st.write("Finally we need to determine last matchday to start the betting cycle. This can be viewed at a stopp loss for the end of the season. At the this point no new betting cycle will start.  Ongoing cycles will be continued however. Of course it is also possible to contineu a cycle over the season final.")
            st.title('APPLY STRATEGY')
            st.write("")
            st.write("")

            initial_stake = st.select_slider('Select initial stake:', options=[i * 0.5 for i in range(1, 41)], key='initial_stake_slider')
            max_bets = st.select_slider('Select max number of bets:', options=list(range(2, 21)), key='max_bets_slider')
            dont_start_func = st.select_slider('Select last matchday to start:', options=list(range(1, 35)), key='dont_start_slider')

            st.write("")
            betting_result = betting_strategy(team_df_selected_season, initial_stake=initial_stake, max_bets=max_bets, dont_start_func=dont_start_func)

            if betting_result:
                st.success(f"Betting strategy for {selected_team} in Season {selected_season} was successful!")
            else:
                st.warning(f"Betting strategy for {selected_team} in Season {selected_season} was not successful!")

            st.subheader("Additional Data:")


        col20, col21 = st.columns([1.7, 1.3])

        with col20:
            plot_non_draw_streaks_for_teams(non_draw_lengths_for_teams, [selected_team])

        with col21:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("It is helpful to analyze the Data to determine the lenght and consistency of draw streaks. Unlike from a fan point of view a inconsistent performance is quite a good sign for the betting strategy.High win or loss streak can risky.")
            st.write("")
            st.write("It is also important to identify pattern of long draw streaks over time and compare the data with other teams.")
          

        col11, col12 = st.columns([2, 1])

        with col11:
            plot = plot_season_results_stacked(team_df_selected_season, selected_team, selected_season)

        with col12:
            columns_to_display = ['Matchday', 'HomeTeam', 'AwayTeam', 'Quote 1', 'Quote 0',
                                  'Quote 2', 'Full Time Result', 'Return']
            st.write("")
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'> Results and Historical Odds {selected_team} in Season {selected_season}:</span>", unsafe_allow_html=True)
            display_team_data(team_df_selected_season, selected_team, columns_to_display)


            st.write("")
            st.write("")

        
        col101, col202, col303 = st.columns(3)

        teams_group1 = team_names[:6]
        teams_group2 = team_names[6:12]
        teams_group3 = team_names[12:]



        with col101:
            st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Non-Draw Streaks for all teams in four years:</span>", unsafe_allow_html=True)

            st.write("")
            for team_name in teams_group1:
                non_draw_streaks_gt_one = calculate_non_draw_lengths_greater_than_one(team_data_frames_for_df[team_name])
                st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>{team_name}: {non_draw_streaks_gt_one}</span>", unsafe_allow_html=True)
        with col202:
            st.write("")
            st.write("")
            st.write("")
            for team_name in teams_group2:
                non_draw_streaks_gt_one = calculate_non_draw_lengths_greater_than_one(team_data_frames_for_df[team_name])
                st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>{team_name}: {non_draw_streaks_gt_one}</span>", unsafe_allow_html=True)

        with col303:
            st.write("")
            st.write("")
            st.write("")
            for team_name in teams_group3:
                non_draw_streaks_gt_one = calculate_non_draw_lengths_greater_than_one(team_data_frames_for_df[team_name])
          
                st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>{team_name}: {non_draw_streaks_gt_one}</span>", unsafe_allow_html=True)
        st.subheader("Over/Under")
        col5, col6 = st.columns([1, 1])
        
        with col5:
       
            plot_over_under_return_vs_target_heatmap(team_df_selected_season, selected_team, selected_season)
          

        with col6:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            st.write("Sometimes the amount of goals scored have also an impact on whether the game result in a draw or not. Normally the if the amount of goals scored is it is high likely to be a non draw.")  
            st.write("")
            st.write("Also a 0:0 happens much more often than a 2:2 or higher. In this sense it could also be helpful to check for some inregularities in over time. Under/Over bets are also pretty popular.")
        
        col3, col4 = st.columns([1, 1])


        with col3:
            st.write("")
            columns_to_display2 = ['Matchday', 'HomeTeam', 'AwayTeam', 'Quote >2.5', 'Quote <2.5', 'O/U Return',
                                   'Home Team Goals', 'Away Team Goals']
            st.write(f"Data (Additional Columns) for {selected_team} in Season {selected_season}:")
            display_team_data(team_df_selected_season, selected_team, columns_to_display2)

        with col4:
            plot_over_under_return_by_matchday_plotly(team_df_selected_season, selected_team, selected_season)
        
        st.write("")
        st.write("")

        col7, col8 = st.columns([1, 1])

        with col7:
            plot_goals_scored_by_matchday_plotly(team_df_selected_season, selected_team, selected_season)

        with col8:
            plot_goals_received_by_matchday_plotly(team_df_selected_season, selected_team, selected_season)

            st.write("")
            st.write("")
            

        col9, col10 = st.columns([2, 1])

        with col9:
            plot_total_goals_by_matchday_plotly(team_df_selected_season, selected_team, selected_season)

        with col10:
            mean_goals_scored = calculate_mean_goals_scored(team_df_selected_season, selected_team)
            mean_goals_received = calculate_mean_goals_received(team_df_selected_season, selected_team)
            mean_total_goals = calculate_mean_total_goals(team_df_selected_season, selected_team)
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.markdown(f"<span style='font-size: 18px; font-weight: bold;'>Mean Goals Scored by {selected_team}: {mean_goals_scored:.1f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 18px; font-weight: bold;'>Mean Goals Received by {selected_team}: {mean_goals_received:.1f}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 18px; font-weight: bold;'>Mean Total Goals by {selected_team}: {mean_total_goals:.1f}</span>", unsafe_allow_html=True)
        st.title("PREDICTIONS:")
        st.write("")

        st.write("The machine learning model predictics possibilities of a non draw streak for every team.Here a higher percentage means a higher changes of a Non Draw Streak. In general the short term streaks are much more likely. However certain teams are an exeption here.")
        st.write("")
        st.write("Here is to say that no long term predictios can be accurate at this point.The predictions for the non-draws streak are solely based on stastical date. In sports team performance, fitness, mental conditions and many other factors playing an important role. This is only to indicate trends")
        st.write("")
        n = st.select_slider("Select Target for Non-Draw Streak:", options=list(range(2, 21)))

        st.markdown(f"<span style='font-size: 16px; font-weight: bold;'>Possibilities for Non Draw Streaks:</span>", unsafe_allow_html=True)

        team_average_probabilities, _ = train_rf_model(team_names, non_draw_lengths_for_teams, team_data_frames_for_df, n)

        colx, coly = st.columns(2)

        with colx:
  
            st.markdown(f"<span style='font-size: 12px; font-weight: bold;'>Possibilities for a Non-Draw Streak</span>", unsafe_allow_html=True)
            for team_name, average_probability in team_average_probabilities.items():
                st.write(f"{team_name}: {average_probability:.2f}%")


        with coly:
            saruman = "/Users/hannesreichelt/Desktop/saruman.png"
            st.image(saruman)  
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("In addition to the long term predicitions I also focused on the first matchday of Season 23/24. For this is event I used a more complex machine learning model and input data for example recent performance, opponent statistics, venues etc.")
            st.write("")
            st.write("However it is not easy to predict a certain match , especially a draw. Nevertheless are statistics a great tools to assist the decision making process.") 
            st.write("")
            st.write("That being said I had a lot of fun working on my first nerdy data science Project.I hope you enjoyed it and maybe be a bit less annoyed when a game ends up in a draw...")
                
    
    else:
        st.write(f"Invalid season selection. Please choose a valid season from the available options.")
else:
    st.write(f"No data available for team: {selected_team}")
st.write("")
st.subheader("Next Matchday:")
st.write("")
Next_Matches = "/Users/hannesreichelt/Desktop/lm.jpg"
st.image(Next_Matches)
team_names = ['Ein Frankfurt', 'Bayern Munich', 'Hertha', 'Augsburg', 'Freiburg', 'Bochum', 'Mainz', "M'gladbach'", 'Hoffenheim', 'Union Berlin', 'Wolfsburg', 'Werder Bremen', 'Dortmund', 'Leverkusen', 'Stuttgart', 'RB Leipzig', 'FC Koln', 'Schalke 04']

for team_name in team_names:
    team_df = team_data_frames_for_df.get(team_name)
    if team_df is not None and not team_df.empty:
        train = team_df[team_df['Date'] < '2023-04-22']
        test = team_df[team_df['Date'] > '2023-04-22']
        
        features = ['week_codes', 'hour_codes', 'venue_codes', 'opponent_codes']
        
        rf = RandomForestClassifier()  # Initialize the RandomForestClassifier
        rf.fit(train[features], train['target'])
        
        preds = rf.predict(test[features])
        
        acc = accuracy_score(test['target'], preds)
        print(f"Accuracy for {team_name}: {acc}")

        combined = pd.DataFrame({'actual': test['target'], 'predictions': preds})
        crosstab_result = pd.crosstab(index=combined['actual'], columns=combined['predictions'])
        print(crosstab_result)

        # Save the trained model for the current team
        model_path = '/Users/hannesreichelt/Desktop/spiced_projects/poisson-ivy-student-code/final_project.joblib'
        joblib.dump(rf, model_path)

# Load the trained model
model_path = '/Users/hannesreichelt/Desktop/spiced_projects/poisson-ivy-student-code/final_project.joblib'
rf = joblib.load(model_path)

venue_mapping = {
    "Werder Bremen": 1,
    "Bayern Munich": 0,
    "Augsburg": 1,
    "Hoffenheim": 1,
    "Stuttgart": 1,
    "Wolfsburg": 0,
    "Dortmund": 1,
    "Union Berlin": 1,
    "Ein Frankfurt": 1,
    "Freiburg": 0,
    "Bochum": 1,
    "RB Leipzig": 0,
    "Leverkusen": 0,
    "M'gladbach": 0,
    "FC Koln": 1,
    "Mainz": 1,
    "Hertha": 0,
    "Schalke 04": 0
}

# Add a custom test match with a draw outcome
next_matchday_data = {
    'week_codes': [4, 5, 5, 5, 5, 5, 5, 6, 6, 4],  # Replace with the actual week codes for the matches
    'hour_codes': [20, 15, 15, 15, 15, 15, 18, 15, 17, 20],  # Replace with the actual hour codes for the matches
    'venue_codes': [venue_mapping["Werder Bremen"], venue_mapping["Bayern Munich"], 
                    venue_mapping["Augsburg"], venue_mapping["Hoffenheim"], 
                    venue_mapping["Stuttgart"], venue_mapping["Wolfsburg"], 
                    venue_mapping["Dortmund"], venue_mapping["Union Berlin"],
                    venue_mapping["Ein Frankfurt"], 1],
    'opponent_codes': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # Replace with the correct opponent codes
}

next_matchday_df = pd.DataFrame(next_matchday_data)

next_matchday_predictions = rf.predict(next_matchday_df)

matchday_schedule = [
    "Werder Bremen vs. Bayern Munich",
    "Leverkusen vs. RB Leipzig",
    "Augsburg vs. 'M'gladbach'",
    "Hoffenheim vs. Freiburg",
    "Stuttgart vs. Bochum",
    "Wolfsburg vs. Heidenheim",
    "Borussia Dortmund vs. FC Köln",
    "Union Berlin vs. Mainz",
    "Ein Frankfurt vs. Darmstadt",
]

for match, prediction in zip(matchday_schedule, next_matchday_predictions):
    outcome = "Draw" if prediction == 1 else "Not Draw"
    st.write(f"{match}: Predicted Outcome - {outcome}")

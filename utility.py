import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

#### WE CAN PUT ANY USEFUL FUNCTIONS IN HERE ####

def make_big_df(start_year=2021, end_year=2023, folder="play-data"):
    big_df = pd.DataFrame()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        df = pd.read_csv(path)
        big_df = big_df._append(df, ignore_index=True)

    
    return big_df

def process(df):
    

    # Filter the DataFrame
    keep_play_types = ['Rush', 'Pass', 'Rushing Touchdown', 'Pass Reception', 'Pass Incompletion',
                            'Passing Touchdown', 'Pass Interception Return', 'Punt', 'Interception', 
                            'Field Goal Good', 'Field Goal Missed']
    df = df[df['Play Type'].isin(keep_play_types)]

    # Simplify play types
    play_type_mapping = {
        'Rushing Touchdown': 'Rush',
        'Pass Reception': 'Pass',
        'Pass Incompletion': 'Pass',
        'Passing Touchdown': 'Pass',
        'Pass Interception Return': 'Pass',
        'Interception': 'Pass',
        'Field Goal Good': 'Field Goal',
        'Field Goal Missed': 'Field Goal'
    }
    df = df.replace(play_type_mapping)

    # Add some useful variables
    df['run'] = np.where(df['Play Type'].str.contains('Rush'), 1, 0)
    df['pass'] = np.where(df['Play Type'].str.contains('Pass'), 1, 0)
    df['Scoring'] = df['Scoring'].astype(int)
    df['totalseconds'] = (df['Clock Minutes'] * 60) + df['Clock Seconds']
    df[['Clock Minutes', 'Clock Seconds']]
    df[['Clock Minutes', 'Clock Seconds', 'totalseconds']]
    df['pointsscored'] = df['Offense Score'] + df['Defense Score']
    df[['Offense Score', 'Defense Score', 'pointsscored']]
    df['pointsscored'] = df['Offense Score'] + df['Defense Score']
    
    # Make lagged variables
    df['L1 Yards Gained'] = df.groupby('Game Id')['Yards Gained'].shift()
    df['L2 Yards Gained'] = df.groupby('Game Id')['Yards Gained'].shift(2)
    df['L1 Play Type'] = df.groupby('Game Id')['Play Type'].shift()
    df['L2 Play Type'] = df.groupby('Game Id')['Play Type'].shift(2)
    df['L1 Down'] = df.groupby('Game Id')['Down'].shift()
    df['L2 Down'] = df.groupby('Game Id')['Down'].shift(2)
    df['L1 Distance'] = df.groupby('Game Id')['Distance'].shift()
    df['L2 Distance'] = df.groupby('Game Id')['Distance'].shift(2)

    key_features = ['Offense Score', 'Defense Score', 'Drive Number', 'Play Number', 'Period', 'totalseconds', 
                'Offense Timeouts', 'Yard Line', 'Yards To Goal', 'Down', 'Distance', 'L1 Play Yards', 'L2 Play Yards',
                'Play Type', 'L1 Yards Gained', 'L2 Yards Gained', 'L1 Play Type', 'L2 Play Type', 'L1 Down', 
                'L2 Down', 'L1 Distance', 'L1 Distance']
    
    # Encode categorical columns
    columns_to_encode = ['Play Type', 'L1 Play Type', 'L2 Play Type']
    label_encoder = LabelEncoder()
    
    df[columns_to_encode] = df[columns_to_encode].apply(lambda col: label_encoder.fit_transform(col))


    return df[key_features]

def readData():
    # Make an array of the DataFrames made from all the CSV files
    dataFrames = []
    # Make an array of the names of the CSV files
    for i in range(1, 10):
        dataFrames.append(pd.read_csv('play-data/2022_wk' + str(i) + '.csv'))

    return dataFrames




def main():
    df = make_big_df()
    # df = pd.DataFrame()
    # df2 = pd.read_csv('byu-football-play-predictions/play-data/2022_wk1.csv')
    # df = df._append(df2)
    print(df)





if __name__== "__main__":
    main()

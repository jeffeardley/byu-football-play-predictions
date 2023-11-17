import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from THEMODEL import THEMODEL
from readData import readData
from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    dataFrames = readData()
    print("Preparing for prediction...")
    # Train the model
    # model = THEMODEL()
    # model.train(dataFrames)

    # Load the trained model (Current Version already saved)
    saved_data = joblib.load('trained_model.joblib')
    trained_model = saved_data['model']
    encoder = saved_data['encoder']
    categorical_columns = saved_data['categorical_columns']
    label_encoder = saved_data['label_encoder']

    ###########################################################
    ############## PREDICTING ON NEW DATA #####################
    ######## HIGHEST ACCURACY SO FAR: 0.6385542168674698#######
    ###########################################################

    # Select the DataFrame for prediction
    X_new = dataFrames[7].drop(['Play Type', 'Id', 'Offense', 'Offense Conference'], axis=1)
    keep_play_types = ['Rush', 'Pass', 'Rushing Touchdown', 'Pass Reception', 'Pass Incompletion',
                       'Passing Touchdown', 'Pass Interception Return']

    play_type_mapping = {
        'Rushing Touchdown': 'Rush',
        'Pass Reception': 'Pass',
        'Pass Incompletion': 'Pass',
        'Passing Touchdown': 'Pass',
        'Pass Interception Return': 'Pass'
    }

    # Handle missing values
    X_new = X_new.fillna(0)

    X_new_encoded = encoder.transform(X_new[categorical_columns])

    predictions = trained_model.predict(X_new_encoded)
    predictions = label_encoder.inverse_transform(predictions)
    # print(predictions)

    y = dataFrames[7]['Play Type']
    y = y.fillna(0)
    y = y[y.isin(keep_play_types)].replace(play_type_mapping)
    # print(y)


if __name__ == '__main__':
    main()

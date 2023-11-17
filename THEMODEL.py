import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import dump, load


class THEMODEL:
    def __init__(self):
        self.categorical_columns = None
        self.model = None
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()

    def train(self, dataFrames):
        combined_df = pd.concat(dataFrames, ignore_index=True)

        # Define play types to keep
        keep_play_types = ['Rush', 'Pass', 'Rushing Touchdown', 'Pass Reception', 'Pass Incompletion',
                           'Passing Touchdown', 'Pass Interception Return']

        # Filter the DataFrame
        combined_df = combined_df[combined_df['Play Type'].isin(keep_play_types)]

        play_type_mapping = {
            'Rushing Touchdown': 'Rush',
            'Pass Reception': 'Pass',
            'Pass Incompletion': 'Pass',
            'Passing Touchdown': 'Pass',
            'Pass Interception Return': 'Pass'
        }

        combined_df = combined_df.replace(play_type_mapping)

        X = combined_df.drop(['Play Type', 'Id', 'Offense', 'Offense Conference'], axis=1)
        y = combined_df['Play Type']

        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in self.categorical_columns:
            X[col] = X[col].astype(str)

        # Handle NaN values
        X = X.fillna(0)
        y = y.fillna(0)

        # One hot encode all the categorical data
        le = LabelEncoder()
        self.encoder.fit(X[self.categorical_columns])
        X_encoded = self.encoder.transform(X[self.categorical_columns])

        y = le.fit_transform(y)

        self.label_encoder = le

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3)

        # self.perform_grid_search(X_train, y_train)

        self.model = MLPClassifier(activation='logistic', alpha=1e-05, beta_1=0.95,
                                   epsilon=1e-08, hidden_layer_sizes=100,
                                   learning_rate='constant', learning_rate_init=0.05,
                                   max_iter=10000, momentum=0.9, random_state=1,
                                   solver='adam')

        self.model.fit(X_train, y_train)

        joblib.dump({'model': self.model, 'encoder': self.encoder, 'categorical_columns': self.categorical_columns, 'label_encoder': self.label_encoder}, 'trained_model.joblib')

        print(self.model.score(X_test, y_test))

    def perform_grid_search(self, X, y):
        parameter_space = {
            'hidden_layer_sizes': [(60, 30), (100,), (50, 100, 50), (100, 100)],
            'alpha': [1e-5, 1e-6, 1e-8],
            'solver': ['sgd', 'adam'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
            'max_iter': [1000, 10000],
            'momentum': [0.9, 0.95, 0.99],
            'beta_1': [0.9, 0.95],
            'epsilon': [1e-8, 1e-6]
        }

        mlp = MLPClassifier(random_state=1)

        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        clf.fit(X, y)

        # Best parameter set
        print('Best parameters found:\n', clf.best_params_)

        # Update the model with best parameters
        self.model = clf.best_estimator_

    def predict(self, X):
        X = X.reindex(columns=self.columns).fillna(0)
        self.model = load('model.joblib')
        return self.model.predict(X)

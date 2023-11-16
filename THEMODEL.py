import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

class THEMODEL:
    def __init__(self):
        self.model = None

    def train(self, dataFrames):
        X = dataFrames[0].drop(['Play Type', 'Id', 'Offense', 'Offense Conference'], axis=1)
        y = dataFrames[0]['Play Type']

        # Handle NaN values
        X = X.fillna(0)
        y = y.fillna(0)

        # One hot encode all of the categorical data
        le = LabelEncoder()
        X = pd.get_dummies(X)
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        self.perform_grid_search(X_train, y_train)

        self.model.fit(X_train, y_train)
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
        return self.model.predict(X)

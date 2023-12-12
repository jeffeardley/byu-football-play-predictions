import utility as ut
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

df = ut.make_big_df()
df = ut.process(df)
restricted_features = ['Offense Score', 'Defense Score', 'Drive Number', 'Play Number', 'Period', 'totalseconds', 
                'Offense Timeouts', 'Yards To Goal', 'Down', 'Distance', 
                'Play Type']
df = df[restricted_features]
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=['Play Type']).to_numpy()
y = df['Play Type'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# times = []
hiddenlayer_vals = [1, 3, 5, 10, 15]
precisions = []
recalls = []
f1 = []
acc = []

for i in hiddenlayer_vals:
    print(i, 'th iteration')
    hidden_layers = tuple([len(X[0])] * i)
    # make model
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='logistic', solver='adam',
                        alpha=.001, batch_size=1, learning_rate_init=.01, shuffle=True,
                        momentum=0, n_iter_no_change=50, max_iter=1000)
    # start_time = time.time()
    mlp.fit(X_train, y_train)
    # end_time = time.time()
    # train_time = end_time - start_time
    # times.append(train_time)

    y_pred = mlp.predict(X)
    mlp_precision = precision_score(y, y_pred, average="weighted")
    mlp_recall = recall_score(y, y_pred, average="weighted")
    mlp_f1 = f1_score(y, y_pred, average="weighted")

    # training set acc
    train_pred = mlp.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_acc = mlp.score(X_train, y_train)

    # test set acc
    test_pred = mlp.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_acc = mlp.score(X_test, y_test)

    print(f'MLP Precision: {mlp_precision}')
    print(f'MLP Recall: {mlp_recall}')
    print(f'MLP F1 Score: {mlp_f1}')
    print(f'MLP Acc: {test_acc}')

    precisions.append(mlp_precision)
    recalls.append(mlp_recall)
    f1.append(mlp_f1)
    acc.append(test_acc)

plt.plot(hiddenlayer_vals, acc, label='Accuracy', marker='o')
plt.plot(hiddenlayer_vals, precisions, label='Precision', marker='o')
plt.plot(hiddenlayer_vals, recalls, label='Recall', marker='o')
plt.plot(hiddenlayer_vals, f1, label='F1', marker='o')

plt.xlabel('Number of Hidden Layers')
plt.ylabel('Accuracy Metric')
plt.title('Performance Metrics vs. Number of Hidden Layers')

plt.legend()

plt.show()


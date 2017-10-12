import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from data_utils import load_image_data


def determine_model(model, X, y_true):
    print()
    print('{:#^40}'.format(' {} '.format(type(model).__name__)))
    y_pred = cross_val_predict(model, X, y_true, cv=3)
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('{:15}'.format('Precision:'), precision_score(y_true, y_pred, average=None))
    print('{:15}'.format('Recall:'), recall_score(y_true, y_pred, average=None))
    print('{:15}'.format('f1 score:'), f1_score(y_true, y_pred, average=None))


if __name__ == '__main__':
    data, labels = load_image_data()
    X_train, y_train, X_test, y_test = split_train_test(data, labels)

    ss = StandardScaler()
    X_train_prepared = ss.fit_transform(X_train.astype(float))

    for m in [RandomForestClassifier(), SGDClassifier()]:
        determine_model(m, X_train_prepared, y_train)
        m.fit(X_train_prepared, y_train)
        y_test_pred = m.predict(ss.transform(X_test.astype(float)))
        print(confusion_matrix(y_test, y_test_pred))

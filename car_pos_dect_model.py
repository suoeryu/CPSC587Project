from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import load_image_data, create_index_csv

if __name__ == '__main__':
    create_index_csv()
    data, labels = load_image_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.6)

    pipe_line = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipe_line.fit_transform(X_train.astype(float), y_train)

    y_test_pred = pipe_line.predict(X_test.astype(float))
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(pipe_line, 'car_pos_detection.pkl')



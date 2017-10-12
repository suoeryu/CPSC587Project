from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import load_image_data, split_train_test
from img_utils import process_image

if __name__ == '__main__':
    data, labels = load_image_data()
    X_train, y_train, X_test, y_test = split_train_test(data, labels, 0.6)

    pipe_line = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipe_line.fit_transform(X_train.astype(float), y_train)

    y_test_pred = pipe_line.predict(X_test.astype(float))
    print(confusion_matrix(y_test, y_test_pred))

    joblib.dump(pipe_line, 'car_detection.pkl')



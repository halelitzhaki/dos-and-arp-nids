import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import joblib


def open_files(model_path, dataset_path):
    loaded_model = joblib.load(model_path)
    print("Model loaded successfully!")
    loaded_dataset = pd.read_csv(dataset_path, header=None)
    print("Dataset loaded successfully!")
    return loaded_model, loaded_dataset


if __name__ == '__main__':
    model, dataset = open_files('nids_model.pkl', 'test_data.csv')
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    y_pred = model.predict(X)

    print(f'Confusion Matrix: \n{confusion_matrix(y, y_pred)}')

    print(f'Accuracy: {accuracy_score(y, y_pred)}')

    print(f"F1 Score: {f1_score(y, y_pred, average='weighted')}")

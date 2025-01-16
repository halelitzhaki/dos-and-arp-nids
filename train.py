import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def categorize_attack(label):
    dos_attacks = [1, 2, 3, 4, 5, 6, 19, 20]  # DoS Attacks
    arp_attacks = [8, 9, 10, 14, 15, 16, 17]  # ARP MitM Attacks
    if label in dos_attacks:
        return "DoS"
    elif label in arp_attacks:
        return "ARP MitM"
    else:
        return "Other"


# Function to load and preprocess data
def load_and_preprocess_data(train_path, test_path):
    train_dataset = pd.read_csv(train_path, header=None)
    test_dataset = pd.read_csv(test_path, header=None)
    train_dataset['attack_category'] = train_dataset.iloc[:, -1].apply(categorize_attack)
    test_dataset['attack_category'] = test_dataset.iloc[:, -1].apply(categorize_attack)

    return train_dataset, test_dataset


# Function to encode labels
def encode_labels(train_data, test_data):
    categorical_columns = [1, 2, 3]  # protocol_type, service, flag
    le = LabelEncoder()
    for col in categorical_columns:
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.transform(test_data[col])

    y_train = le.fit_transform(train_data['attack_category'])
    y_test = le.transform(test_data['attack_category'])

    X_train = train_data.drop(columns=[41, 'attack_category'])
    X_test = test_data.drop(columns=[41, 'attack_category'])

    return X_train, X_test, y_train, y_test, le


# Function to scale features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_random_forest(X_train_scaled, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    return rf_model

def train_grid_search(X_train_scaled, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    y_pred_optimized = grid_search.best_estimator_.predict(X_test_scaled)
    print("Optimized Model - Classification Report:\n",
          classification_report(y_test, y_pred_optimized, target_names=le.classes_))
    print("Optimized Accuracy:", accuracy_score(y_test, y_pred_optimized))

# Function to evaluate the model
def evaluate_model(model, X_test_scaled, y_test, x_train_scaled, y_train, le):
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator=model, X=x_train_scaled, y=y_train, cv=5)
    print('Performance on the validation set: Cross Validation Score = %0.4f' % accuracies.mean())


# Function to visualize attack type distribution
def plot_attack_distribution(train_data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_data)
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# Function to compare predictions with true labels
def compare_predictions(rf_model, X_test_scaled, y_test, le):
    y_pred = rf_model.predict(X_test_scaled)
    comparison_df = pd.DataFrame({
        'True Label': le.inverse_transform(y_test),
        'Predicted Label': le.inverse_transform(y_pred)
    })
    print("\n\nExamples of Predictions vs True Labels:")
    print(comparison_df.sample(10))


# Function to analyze predictions and true labels distribution
def analyze_predictions_distribution(y_pred, y_test, le):
    pred_counts = pd.Series(le.inverse_transform(y_pred)).value_counts()
    true_counts = pd.Series(le.inverse_transform(y_test)).value_counts()
    print("Prediction Distribution:\n", pred_counts)
    print("\nTrue Label Distribution:\n", true_counts)


# Main function to execute the workflow
if __name__ == '__main__':
    train_dataset_file_path = 'Datasets/KDDTrain+.txt'
    test_dataset_file_path = 'Datasets/KDDTest+.txt'

    # Step 1: Load and preprocess data
    train_data, test_data = load_and_preprocess_data(train_dataset_file_path, test_dataset_file_path)

    # Step 2: Encode labels
    X_train, X_test, y_train, y_test, le = encode_labels(train_data, test_data)

    # Step 3: Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Step 4: Train the model
    rf_model = train_random_forest(X_train_scaled, y_train)

    # Step 5: Evaluate the model
    evaluate_model(rf_model, X_test_scaled, y_test, X_train_scaled, y_train, le)

    # Step 6: Visualize attack type distribution
    plot_attack_distribution(train_data['attack_category'])

    # Step 7: Compare predictions with true labels
    compare_predictions(rf_model, X_test_scaled, y_test, le)

    # Step 8: Analyze predictions and true labels distribution
    y_pred = rf_model.predict(X_test_scaled)
    analyze_predictions_distribution(y_pred, y_test, le)



#
# def models(x_train, x_test, y_train, y_test):
#     # Random Forest
#     rf_model = RandomForestClassifier(random_state=42)
#     print_scores("Random Forest", fit_and_evaluate(rf_model, x_train, x_test, y_train),
#                  y_test, rf_model.predict(x_test))
#
#     # Other..
#
#
# def fit_and_evaluate(model, x_train, x_test, y_train):
#     model.fit(x_train, y_train)
#     model_pred = model.predict(x_test)
#     model_cross = cross_val(x_train, y_train, model)
#     return model_cross
#
#
# def print_scores(model_name, cross, y_test, y_pred):
#     print(f'{model_name} Performance on the validation set: Cross Validation Score = %0.4f' % cross)
#     # Make predictions
#     # Print the accuracy
#     print("Accuracy on test dataset: ", accuracy_score(y_test, y_pred))
#
#     # Calculate and print the F1 score
#     f1 = f1_score(y_test, y_pred, average='weighted')  # Adjust 'average' as needed
#     print(f"\nF1 Score on test dataset: {f1}")
#
#
# def cross_val(x_train, y_train, model):
#     accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv=5)
#     return accuracies.mean()

# train_grid_search(X_train_scaled, y_train)
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


def categorize_attack(label) -> str:
    """ Classifies the label if it's DoS/ ARP MitM/ Other, based on the attack number identifier. """
    dos_attacks = [1, 2, 3, 4, 5, 6, 19, 20]  # DoS Attacks
    arp_attacks = [8, 9, 10, 14, 15, 16, 17]  # ARP MitM Attacks
    if label in dos_attacks:
        return "DoS"
    elif label in arp_attacks:
        return "ARP MitM"
    else:
        return "Other"


def load_and_preprocess_data(train_path, test_path) -> []:
    """ Load the train and test datasets and preprocess them. """
    train_dataset = pd.read_csv(train_path, header=None)
    test_dataset = pd.read_csv(test_path, header=None)

    # Adding the features names to the datasets
    columns = (['duration'
        , 'protocol_type'
        , 'service'
        , 'flag'
        , 'src_bytes'
        , 'dst_bytes'
        , 'land'
        , 'wrong_fragment'
        , 'urgent'
        , 'hot'
        , 'num_failed_logins'
        , 'logged_in'
        , 'num_compromised'
        , 'root_shell'
        , 'su_attempted'
        , 'num_root'
        , 'num_file_creations'
        , 'num_shells'
        , 'num_access_files'
        , 'num_outbound_cmds'
        , 'is_host_login'
        , 'is_guest_login'
        , 'count'
        , 'srv_count'
        , 'serror_rate'
        , 'srv_serror_rate'
        , 'rerror_rate'
        , 'srv_rerror_rate'
        , 'same_srv_rate'
        , 'diff_srv_rate'
        , 'srv_diff_host_rate'
        , 'dst_host_count'
        , 'dst_host_srv_count'
        , 'dst_host_same_srv_rate'
        , 'dst_host_diff_srv_rate'
        , 'dst_host_same_src_port_rate'
        , 'dst_host_srv_diff_host_rate'
        , 'dst_host_serror_rate'
        , 'dst_host_srv_serror_rate'
        , 'dst_host_rerror_rate'
        , 'dst_host_srv_rerror_rate'
        , 'attack'
        , 'level'])
    train_dataset.columns = columns
    test_dataset.columns = columns

    train_dataset['attack_category'] = train_dataset.iloc[:, -1].apply(categorize_attack)
    test_dataset['attack_category'] = test_dataset.iloc[:, -1].apply(categorize_attack)

    return train_dataset, test_dataset


def encode_labels(train_data, test_data) -> []:
    """ Encoding the data labels, and splitting them to X and y. """
    categorical_columns = ['protocol_type', 'service', 'flag', 'attack']
    le = LabelEncoder()
    for col in categorical_columns:
        train_data[col] = le.fit_transform(train_data[col])
        test_data[col] = le.fit_transform(test_data[col])

    y_train = le.fit_transform(train_data['attack_category'])
    y_test = le.transform(test_data['attack_category'])

    X_train = train_data.drop(columns=['attack', 'attack_category'], axis=1)
    X_test = test_data.drop(columns=['attack', 'attack_category'], axis=1)

    return X_train, X_test, y_train, y_test, le


def select_features(X_train, y_train, X_test) -> []:
    """ Select the features for the model, with Select-K-Best algorithm. """
    select_model = SelectKBest(mutual_info_classif, k=16)
    select_model.fit(X_train, y_train)
    selected_features = X_train.columns[select_model.get_support()]

    return X_train[selected_features], X_test[selected_features]


def scale_features(X_train, X_test) -> []:
    """ Scaling and normalizing the features' values for the model's input. """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_random_forest(X_train_scaled, y_train) -> RandomForestClassifier:
    """ Creating and training the Random-Forest model. """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    return rf_model


def evaluate_model(model, y_test, x_train_scaled, y_train, le) -> None:
    """" Evaluates the model accuracy with Cross-Validation and Grid-Search. """
    print(f'Cross Validation Score = {cross_val_score(estimator=model, X=x_train_scaled, y=y_train, cv=5).mean()}')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                               scoring="f1_macro", n_jobs=-1, return_train_score=True).fit(X_train_scaled, y_train)
    print(f'Grid Search Cross-Validation Score: {grid_search.best_score_}')


def plot_attack_distribution(train_data) -> None:
    """ Visualizing the attack type distribution. """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_data)
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


def compare_predictions(rf_model, X_test_scaled, y_test, le) -> None:
    """ Comparing the predictions with true labels. """
    y_pred = rf_model.predict(X_test_scaled)
    comparison_df = pd.DataFrame({
        'True Label': le.inverse_transform(y_test),
        'Predicted Label': le.inverse_transform(y_pred)
    })
    print("\n\nExamples of Predictions vs True Labels:")
    print(comparison_df.sample(10))


def analyze_predictions_distribution(y_pred, y_test, le) -> None:
    """ Analyzing the predictions and true labels distribution. """
    pred_counts = pd.Series(le.inverse_transform(y_pred)).value_counts()
    true_counts = pd.Series(le.inverse_transform(y_test)).value_counts()
    print("Prediction Distribution:\n", pred_counts)
    print("\nTrue Label Distribution:\n", true_counts)


def save_files(model, X_test, y_test, features) -> None:
    """ Saves the model to .pkl, and the test's dataset to .csv """
    joblib.dump(model, 'nids_model.pkl')
    print("Model saved successfully!")

    y_test_series = pd.Series(y_test)
    X_test_df = pd.DataFrame(X_test)
    combined = pd.concat([X_test_df, y_test_series], axis=1)
    combined.to_csv('test_data.csv', index=False, header=False)
    print("Dataset for test saved successfully!")


if __name__ == '__main__':
    print("Hello! \nThis script is building the model, and then evaluates it with Cross-Validation and Grid-Search. "
          "\nIt might take a while, please be patient... :)\n")

    train_dataset_file_path = 'Datasets/KDDTrain+.txt'
    test_dataset_file_path = 'Datasets/KDDTest+.txt'

    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(train_dataset_file_path, test_dataset_file_path)

    # Encode labels
    X_train, X_test, y_train, y_test, le = encode_labels(train_data, test_data)

    X_train, X_test = select_features(X_train, y_train, X_test)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train the model
    rf_model = train_random_forest(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(rf_model, y_test, X_train_scaled, y_train, le)

    # Visualize attack type distribution
    plot_attack_distribution(train_data['attack_category'])

    # Compare predictions with true labels
    compare_predictions(rf_model, X_test_scaled, y_test, le)

    # Analyze predictions and true labels distribution
    y_pred = rf_model.predict(X_test_scaled)
    analyze_predictions_distribution(y_pred, y_test, le)

    # Save the model and the test dataset
    save_files(rf_model, X_test_scaled, y_test, X_test.columns)

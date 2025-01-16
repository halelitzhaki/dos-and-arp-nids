import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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


# Function to encode labels
def encode_labels(train_data, test_data):
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


def select_features(X_train, y_train, X_test):
    select_model = SelectKBest(mutual_info_classif, k=30)
    select_model.fit(X_train, y_train)
    selected_features = X_train.columns[select_model.get_support()]

    return X_train[selected_features], X_test[selected_features]


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

    X_train, X_test = select_features(X_train, y_train, X_test)

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

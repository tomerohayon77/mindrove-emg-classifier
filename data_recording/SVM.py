import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#
def load_feature_files(directory):
    all_features = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('features_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Reading file: {file_path}")  # Debugging line
                df = pd.read_csv(file_path)
                all_features.append(df)
    if not all_features:
        raise ValueError("No feature files found in the directory.")
    return pd.concat(all_features, ignore_index=True)

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def regular_train_test_split(features_df):
    # Drop rows with label 5
    features_df = features_df[features_df['Label'] != 5]

    # Separate majority and minority classes
    majority_class = features_df[features_df['Label'] == 0]
    minority_classes = features_df[features_df['Label'] != 0]

    # Determine the sample size for undersampling
    sample_size = min(len(majority_class), len(minority_classes))

    # Undersample the majority class
    majority_class_undersampled = majority_class.sample(sample_size, random_state=42)

    # Combine the undersampled majority class with the minority classes
    balanced_df = pd.concat([majority_class_undersampled, minority_classes])

    X = balanced_df.drop(columns=['Label', 'SessionID'])
    y = balanced_df['Label']

    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_svm(X_train, X_test, y_train, y_test):
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    directory = r'C:\Technion\Project_A\Project_A\Feature files'
    features_df = load_feature_files(directory)
    X_train, X_test, y_train, y_test = regular_train_test_split(features_df)
    train_svm(X_train, X_test, y_train, y_test)
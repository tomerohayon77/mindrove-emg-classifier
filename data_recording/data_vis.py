import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def visualize_data(file_path):
    # Load the processed data
    df = pd.read_csv(file_path)

    # Plot the distribution of labels
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Label', data=df)
    plt.title('Distribution of Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

    # Plot the correlation matrix
    plt.figure(figsize=(20, 18))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Plot the feature distributions for each label
    feature_columns = df.columns[:-2]  # Exclude 'Label' and 'SessionID'
    """ for feature in feature_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Label', y=feature, data=df)
        plt.title(f'Distribution of {feature} by Label')
        plt.xlabel('Label')
        plt.ylabel(feature)
        plt.show()

    # 2D Scatter plots for all pairs of features in columns 0 to 4
    for i in range(5):
        for j in range(i + 1, 5):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature_columns[i], y=feature_columns[j], hue='Label', data=df, palette='Set2')
            plt.title(f'Scatter Plot of {feature_columns[i]} vs {feature_columns[j]}')
            plt.xlabel(feature_columns[i])
            plt.ylabel(feature_columns[j])
            plt.legend(title='Label')
            plt.show()"""

    # Balance the dataset by undersampling the majority class
    df_majority = df[df.Label == 0]
    df_minority = df[df.Label != 0]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,    # sample without replacement
                                       n_samples=len( df_minority),  # to match minority class
                                       random_state=42)  # reproducible results

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Train and evaluate SVM model
    X = df_balanced.drop(columns=['Label', 'SessionID'])
    y = df_balanced['Label']

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    visualize_data(r'C:\Users\User\PycharmProjects\Project_A\EMG\all_the_features\combined_features.csv')
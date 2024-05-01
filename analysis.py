import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

def read_data(test_file, label_file):
    test = pd.read_csv(test_file, header=None)
    test_label = pd.read_csv(label_file, header=None)
    return test, test_label

def plot_time_series(test, test_label):
    plt.figure(figsize=(12, 6))
    test.index = pd.to_numeric(test.index, errors='coerce')
    test.iloc[:, 0] = pd.to_numeric(test.iloc[:, 0], errors='coerce')
    plt.plot(test.index, test.iloc[:, 0], color='blue', label='Test Data')
    plt.scatter(test_label[test_label == 1].index, test[test_label == 1].iloc[:, 0], color='red', label='Anomaly', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Time Series Plot with Anomalies')
    plt.legend()
    plt.show()

def perform_eda(test, test_label):
    corr_matrix = test.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    anomaly_indices = test_label[test_label[0] == 1].index
    anomaly_data = test.loc[anomaly_indices]

    anomaly_scores = []
    for col in anomaly_data.columns:
        anomaly_deviation = abs(anomaly_data[col] - anomaly_data[col].mean())
        avg_abs_deviation = anomaly_deviation.mean()
        anomaly_scores.append(avg_abs_deviation)

    max_score_idx = anomaly_scores.index(max(anomaly_scores))
    max_deviation_col = anomaly_data.columns[max_score_idx]

    for i in range(len(anomaly_scores)):
        if anomaly_scores[i] >= 0.1:
            print(i, end=' ')

    high_deviation_cols = [col for col, score in zip(anomaly_data.columns, anomaly_scores) if score > 0.5]

    if high_deviation_cols:
        for col in high_deviation_cols:
            print(col)

    plt.figure(figsize=(12, 6))
    for column in anomaly_data.columns:
        sns.kdeplot(anomaly_data[column], label='Anomaly', shade=True)
        plt.title('Distribution of Column ' + str(column))
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    decomposition = seasonal_decompose(test.iloc[:, 0], model='additive', period=24)
    decomposition.plot()
    plt.show()

test_file = 'psm_test.csv'
label_file = 'psm_test_label.csv'
test, test_label = read_data(test_file, label_file)
plot_time_series(test, test_label)
perform_eda(test, test_label)

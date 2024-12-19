import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Set the display width
pd.set_option("display.width", 2000)

# Step 2: Load the dataset
# from google.colab import files
# uploaded = files.upload()

# Step 3: Explore the dataset
vr_data = pd.read_csv('D:\CODING UPM\PYTHON\SEM4_UPM\CNS3102_Python\Project\Student_Performance\Student_performance_data _.csv')
print(vr_data.shape)
print(vr_data)
print(vr_data.isnull().sum())

# Drop the rows with missing values
vr_data.dropna(inplace=True)

# Split the data into features and target
X = vr_data.iloc[:, :-1]
y = vr_data.iloc[:, -1]

# Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create a list to store accuracy and time measurements
classifiers = ['DT', 'K-NN', 'SVM','RF']
accuracy_scores = []
training_times = []
testing_times = []

# Define a function to measure time taken for training and testing
def measure_time(classifier, X_train, y_train, X_test):
    start_train = time.time()
    classifier.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    
    start_test = time.time()
    classifier.predict(X_test)
    end_test = time.time()
    test_time = end_test - start_test
    
    return train_time, test_time

# Create and train the classifiers

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_train_time, dt_test_time = measure_time(dt_classifier, X_train, y_train, X_test)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
accuracy_scores.append(dt_accuracy)
training_times.append(dt_train_time)
testing_times.append(dt_test_time)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_train_time, knn_test_time = measure_time(knn_classifier, X_train, y_train, X_test)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
accuracy_scores.append(knn_accuracy)
training_times.append(knn_train_time)
testing_times.append(knn_test_time)

# Support Vector Machines Classifier
svm_classifier = SVC()
svm_train_time, svm_test_time = measure_time(svm_classifier, X_train, y_train, X_test)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
accuracy_scores.append(svm_accuracy)
training_times.append(svm_train_time)
testing_times.append(svm_test_time)

# Support Vector Machines Classifier
svm_classifier = RandomForestClassifier()
svm_train_time, svm_test_time = measure_time(svm_classifier, X_train, y_train, X_test)
svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
accuracy_scores.append(svm_accuracy)
training_times.append(svm_train_time)
testing_times.append(svm_test_time)

# Create a comparison table
data = {'Classifier': classifiers, 'Accuracy': accuracy_scores, 'Training Time': training_times, 'Testing Time': testing_times}
comparison_df = pd.DataFrame(data)
print(comparison_df)

# Create comparison graphs
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Bar chart for accuracy comparison
axs[0].bar(classifiers, accuracy_scores)
axs[0].set_xlabel('Classifier')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy Comparison')

# Bar chart for training and testing time comparison
bar_width = 0.3
index = np.arange(len(classifiers))

axs[1].bar(index, training_times, bar_width, label='Training Time')
axs[1].bar(index + bar_width, testing_times, bar_width, label='Testing Time')
axs[1].set_xlabel('Classifier')
axs[1].set_ylabel('Time (seconds)')
axs[1].set_title('Training and Testing Time Comparison')
axs[1].set_xticks(index + bar_width / 2)
axs[1].set_xticklabels(classifiers)
axs[1].legend()

plt.tight_layout()
plt.show()
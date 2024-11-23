
import pandas as pd
import os
import re
import numpy as np
from collections import Counter
from sklearn.svm import SVC

#############################################################################################################################

### This code is used to create the test dataset in the form of text files ###

# Read the CSV dataset and drop unwanted columns from the file
master_df = pd.read_csv("train dataset.csv", encoding="ISO-8859-1")

master_df_test = pd.DataFrame(master_df).iloc[5000:].reset_index(drop = True, inplace = False).rename(columns={"Category":"label", "Message":"message"})

if not os.path.exists("test"):
    os.makedirs("test")
    
for index, row in master_df_test.iterrows():

    row_data = str(row[1])
    filename = os.path.join("test", f"email{index+1}.txt")
    with open(filename, "w", encoding = "ISO-8859-1") as f:
        f.write(row_data)

y_true = master_df_test["label"]
y_true.to_csv("y_true.csv", index=False)

#############################################################################################################################

### Actual code ###

# Read the CSV dataset and drop unwanted columns from the file
master_df = pd.read_csv("train dataset.csv", encoding="ISO-8859-1")

dataset = pd.DataFrame(master_df).iloc[0:5000].reset_index(drop = True, inplace = False).rename(columns={"Category":"label", "Message":"message"})

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def accuracy_score(y_true, y_pred):
    pred = np.sum(y_true == y_pred)
    acc = pred / len(y_true)
    return acc

# Naive Bayes Classifier from Scratch
class NaiveBayesClassifier:
    def __init__(self):
        self.word_probs = {}
        self.class_probs = {}

    def fit(self, X, y):
        # Calculate prior probabilities
        self.class_probs[1] = np.mean(y == 1)  # Probability of spam
        self.class_probs[0] = np.mean(y == 0)  # Probability of ham
        
        # Separate spam and ham messages
        spam_messages = X[y == 1]
        ham_messages = X[y == 0]
        
        # Calculate total word counts for each class
        total_spam_words = np.sum(spam_messages)
        total_ham_words = np.sum(ham_messages)
        
        # Calculate word probabilities given each class
        self.word_probs[1] = (np.sum(spam_messages, axis=0) + 1) / (total_spam_words + X.shape[1])
        self.word_probs[0] = (np.sum(ham_messages, axis=0) + 1) / (total_ham_words + X.shape[1])

    def predict(self, X):
        predictions = []
        for message in X:
            # Calculate log-probabilities for spam and ham
            log_spam_prob = np.log(self.class_probs[1])
            log_ham_prob = np.log(self.class_probs[0])
            for i, word_count in enumerate(message):
                if word_count > 0:  # Only consider words that appear in the message
                    log_spam_prob += word_count * np.log(self.word_probs[1][i])
                    log_ham_prob += word_count * np.log(self.word_probs[0][i])
            # Choose the class with higher log-probability
            predictions.append(1 if log_spam_prob > log_ham_prob else 0)
        return np.array(predictions)
    
class KNearestNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # Calculate Euclidean distance
            k_nearest_indices = distances.argsort()[:self.k]  # Get indices of k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]  # Get the labels of those neighbors
            majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]  # Majority vote
            predictions.append(majority_vote)
        return np.array(predictions)
    
class LogisticRegressionClassifier:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            # Gradient descent updates
            dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(y)) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]

dataset['message'] = dataset['message'].apply(preprocess_text)
# Convert labels to binary format (1 for spam, 0 for ham)
dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})

# Separate the dataset into spam and ham
spam_data = dataset[dataset['label'] == 1]
ham_data = dataset[dataset['label'] == 0]

################################################ OVERSAMPLING #############################################

# Oversample spam data to match the size of ham data
oversampled_spam_data = spam_data.sample(len(ham_data), replace=True)
balanced_data = pd.concat([ham_data, oversampled_spam_data])

# Shuffle the balanced dataset  
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract messages and labels
X_train = balanced_data['message']
y_train = balanced_data['label'].values

###########################################################################################################


################### This code is used to obtain the test dataset from the text files ######################

folder_path = "test" 
data = []

if not os.path.exists("test"):
    print("test folder does not exist")
    
if not os.path.exists(folder_path):
    print(f"{folder_path} folder does not exist")
else:
    # Sort filenames numerically by extracting the numeric part
    filenames = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0].replace('email', '')))
    
    # Loop through each sorted file
    for filename in filenames:
        if filename.endswith(".txt"):  # Check for .txt files
            file_path = os.path.join(folder_path, filename)  # Full path to the file
            
            # Read the file content
            with open(file_path, "r", encoding="ISO-8859-1") as file:
                content = file.read().strip()  # Read and strip whitespace/newlines
                
            # Append the content to the list
            data.append({"message": content})

###########################################################################################################

# Create a DataFrame from the list
master_df_test = pd.DataFrame(data)

X_test = master_df_test['message']
X_test = X_test.apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

### Training the models ###

print("Training Naive Bayes Classifier...")
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)
print("Naive Bayes Classifier trained succesfully")

print("Training SVM Classifier...")
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
print("SVM Classifier trained succesfully")

print("Training KNN Classifier...")
knn_classifier = KNearestNeighborsClassifier(k=3)
knn_classifier.fit(X_train, y_train)
print("KNN Classifier trained succesfully")

print("Training Logistic Regression Classifier...")
logistic_classifier = LogisticRegressionClassifier(lr=0.01, epochs=1000)
logistic_classifier.fit(X_train, y_train)
print("Logistic Regression Classifier trained succesfully")

### Testing the models ####

print("Testing models...")
nb_predictions = nb_classifier.predict(X_test)
svc_predictions = svc_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)
logistic_predictions = logistic_classifier.predict(X_test)
print("Models tested")

# Mapping predictions to 'spam' and 'ham' labels
nb_outputs = ["spam" if pred == 1 else "ham" for pred in nb_classifier.predict(X_test)]
svc_outputs = ["spam" if pred == 1 else "ham" for pred in svc_classifier.predict(X_test)]
knn_outputs = ["spam" if pred == 1 else "ham" for pred in knn_classifier.predict(X_test)]
logistic_outputs = ["spam" if pred == 1 else "ham" for pred in logistic_classifier.predict(X_test)]

# Outputting the results to a .csv file
results = {
    'Naive Bayes': nb_outputs,
    'SVM': svc_outputs,
    'KNN': knn_outputs,
    'Logistic Regression': logistic_outputs
}
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)

# Taking true y values from .csv file

y_test = pd.read_csv("y_true.csv", header=None, skiprows=1)
y_test = y_test[0].map({'spam': 1, 'ham': 0})
y_test=y_test.values

# Calculating accuracy

nb_accuracy = accuracy_score(y_test, nb_predictions)
svc_accuracy = accuracy_score(y_test, svc_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)  
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")
print(f"SVM Accuracy: {svc_accuracy:.2f}")
print(f"KNN Accuracy: {knn_accuracy:.2f}")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
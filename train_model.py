import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Random Forest
model = RandomForestClassifier()
model.fit(x_train, y_train)
score = accuracy_score(model.predict(x_test), y_test)
print('Random Forest Accuracy: {:.2f}%'.format(score * 100))

# SVM
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_score = accuracy_score(svm_model.predict(x_test), y_test)
print('SVM Accuracy: {:.2f}%'.format(svm_score * 100))

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(x_train, y_train)
gb_score = accuracy_score(gb_model.predict(x_test), y_test)
print('Gradient Boosting Accuracy: {:.2f}%'.format(gb_score * 100))

# Accuracy scores
scores = [score, svm_score, gb_score]
labels = ['Random Forest', 'SVM', 'Gradient Boosting']

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, scores, color=['blue', 'green', 'orange'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

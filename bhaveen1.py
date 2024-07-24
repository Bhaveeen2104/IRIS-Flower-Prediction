import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV file
file_path = 'C:/Users/ASUS/Desktop/python/IRIS Flower.csv'
data = pd.read_csv(file_path)

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target

# Encode target labels if they are not numeric
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Make predictions on the test set
test_predictions = clf.predict(test_data)

# Calculate accuracy
accuracy = accuracy_score(test_target, test_predictions)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Taking input for prediction
d1 = float(input("Enter sepal length : "))
d2 = float(input("Enter sepal width : "))
d3 = float(input("Enter petal length : "))
d4 = float(input("Enter petal width : "))

data_input = np.array([[d1, d2, d3, d4]])

# Predict and decode the result
prediction = clf.predict(data_input)
decoded_prediction = le.inverse_transform(prediction)
print(f'Predicted iris species: {decoded_prediction[0]}')
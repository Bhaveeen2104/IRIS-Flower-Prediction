This project demonstrates the use of a Decision Tree Classifier to predict the species of an Iris flower based on its sepal and petal measurements. The dataset used is the famous Iris dataset, which includes measurements for sepal length, sepal width, petal length, and petal width for three species of Iris flowers: Setosa, Versicolour, and Virginica.

Project Overview
The main steps in this project include:

Loading the Dataset: Using Pandas to read the Iris dataset from a CSV file.
Data Preprocessing:
Splitting the dataset into features (X) and target (y).
Encoding target labels to numeric values with LabelEncoder.
Splitting the Data: Using train_test_split to divide the dataset into training and testing sets.
Training the Model: Training a Decision Tree Classifier on the training data.
Evaluating the Model: Predicting on the test set and calculating the accuracy of the model.
Making Predictions: Taking user input for sepal and petal measurements to predict the Iris species.

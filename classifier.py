# 0. importing required frameworks and tools
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. load data
data = load_iris()
inputs = data.data
results = data.target

print("Dataset: ")
print(inputs)
print(results)

# 2. Instantiate a classifier 
classifier = RandomForestClassifier()

# 3. Split the data(into training and testing data)
X_train, X_test, y_train, y_test = train_test_split(inputs, results, test_size=0.2)

# 4. Train the classifier
fitted = classifier.fit(X_train, y_train)

# 5. Evaluate the classifier: Test the trained model and see how good it is
y_pred = fitted.predict(X_test)

# 6. Accuracy test
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: ", accuracy * 100, "%")

# Detailed report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
 

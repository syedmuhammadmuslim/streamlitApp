import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load the pre-trained model
with open('K-Nearest Neighbors Accuracy.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the test data
test_data = pd.read_csv('mobile_price_range_data.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Generate test predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')

print(f'Accuracy: {accuracy}')
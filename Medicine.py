import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("medical data.csv")

# Select relevant columns and drop rows with missing values
data_col = data[['Disease', 'Medicine']].dropna()

# Encode the 'Disease' and 'Medicine' columns
disease_encoder = LabelEncoder()
medicine_encoder = LabelEncoder()

data_col['disease_encoded'] = disease_encoder.fit_transform(data_col['Disease'])
data_col['medicine_encoded'] = medicine_encoder.fit_transform(data_col['Medicine'])

# Define features and target
X = data_col[['disease_encoded']]
Y = data_col['medicine_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predict the medicine for a new disease
new_disease = 'Common Cold'
new_disease_encoded = disease_encoder.transform([new_disease])[0]
predicted_medicine_encoded = model.predict([[new_disease_encoded]])
predicted_medicine = medicine_encoder.inverse_transform(predicted_medicine_encoded)
print(f'Predicted Medicine for {new_disease}: {predicted_medicine[0]}')

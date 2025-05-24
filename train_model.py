import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Ensure the 'trained_data' directory exists
os.makedirs('trained_data', exist_ok=True)

# Step 1: Load the dataset
df = pd.read_csv('csv/Students_Grading_Dataset.csv')

# Step 2: Clean the dataset (handling missing values if necessary)
# Fill missing values only for numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 3: Convert Grade into a binary pass/fail variable (pass = A, B, C; fail = D, F)
df['pass_fail'] = df['Grade'].apply(lambda x: 1 if x in ['A', 'B', 'C'] else 0)

# Step 4: Prepare the features (X) and target variable (y)
X = df[['Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Attendance (%)', 'Stress_Level (1-10)']]  # Features
y = df['pass_fail']  # Target variable (pass or fail)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the data (optional but recommended for many algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 10: Evaluate the model (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 11: Save the trained model and scaler to files using joblib
joblib.dump(model, 'trained_data/student_pass_model.pkl')  # Save the trained model
joblib.dump(scaler, 'trained_data/scaler.pkl')  # Save the scaler for preprocessing the input data

print("Model and scaler saved successfully!")

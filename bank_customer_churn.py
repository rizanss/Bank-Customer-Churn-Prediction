import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('/content/Customer-Churn-Records.csv')
df.head(10)

# Drop kolom yang tidak diperlukan
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# Cek Dataset
df.head(10)

# Encode kolom kategorikal
label_encoder = LabelEncoder()

df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Card Type'] = label_encoder.fit_transform(df['Card Type'])

# Cek dataset setelah encoding
df.head(10)

# Pisahkan fitur dan target
X = df.drop(columns=['Exited'])
y = df['Exited']

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nData Training: {len(X_train)} sampel")
print(f"Data Testing: {len(X_test)} sampel")

# Model Logistic Regression
logreg = LogisticRegression(max_iter=1000)  # max_iter ditambah untuk konvergensi
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Evaluasi Logistic Regression
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")
print(classification_report(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# Model Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Evaluasi Decision Tree
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(classification_report(y_test, y_pred_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True)
plt.show()

# Contoh inputan pelanggan
test_data = {
    'CreditScore': [400, 800, 600],
    'Geography': ['Spain', 'France', 'Germany'],  # Encode menggunakan LabelEncoder
    'Gender': ['Female', 'Male', 'Female'],       # Encode menggunakan LabelEncoder
    'Age': [60, 30, 40],
    'Tenure': [2, 8, 5],
    'Balance': [0, 150000, 100000],
    'NumOfProducts': [4, 1, 2],
    'HasCrCard': [1, 1, 1],
    'IsActiveMember': [0, 1, 1],
    'EstimatedSalary': [50000, 200000, 100000],
    'Complain': [1, 0, 0],
    'Satisfaction Score': [2, 5, 3],
    'Card Type': ['SILVER', 'PLATINUM', 'GOLD'],  # Encode menggunakan LabelEncoder
    'Point Earned': [100, 900, 500]
}

# Convert to DataFrame
test_df = pd.DataFrame(test_data)

# Encode kolom kategorikal
test_df['Geography'] = label_encoder.fit_transform(test_df['Geography'])
test_df['Gender'] = label_encoder.fit_transform(test_df['Gender'])
test_df['Card Type'] = label_encoder.fit_transform(test_df['Card Type'])

print(test_df)

# Prediksi menggunakan Logistic Regression
logreg_pred = logreg.predict(test_df)
print("Logistic Regression Predictions:", logreg_pred)

# Prediksi menggunakan Decision Tree
tree_pred = tree.predict(test_df)
print("Decision Tree Predictions:", tree_pred)

# Probabilitas churn menggunakan Logistic Regression
logreg_proba = logreg.predict_proba(test_df)
print("Logistic Regression Probabilities:")
print(logreg_proba)

# Probabilitas churn menggunakan Decision Tree
tree_proba = tree.predict_proba(test_df)
print("Decision Tree Probabilities:")
print(tree_proba)

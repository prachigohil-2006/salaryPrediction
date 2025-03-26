import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

path = "../data/Salary_Data.csv"
df = pd.read_csv(path)

df = df.dropna()

X = df[['Years of Experience', 'Education Level', 'Job Title']]
y = df['Salary']

X = pd.get_dummies(X, columns=['Education Level', 'Job Title'])

scaler = StandardScaler()

X[['Years of Experience']] = scaler.fit_transform(X[['Years of Experience']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linearModel = LinearRegression()
linearModel.fit(X_train, y_train)

# score = linearModel.score(X_test, y_test)
# print(f"R2 Score after scaling: {score:.2f}")  == 0.86

os.makedirs('./models', exist_ok=True)

with open('./models/linear_model.pkl', 'wb') as f:
    pickle.dump(linearModel, f)

with open('./models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('./models/one_hot_categories.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)  #save all column names after encoding

# print("model, scaler, and column names saved successfully.")

print("Coefficients:")
for feature, coef in zip(X_train.columns, linearModel.coef_):
    print(f"{feature}: {coef}")

coefficients = pd.DataFrame(linearModel.coef_, X_train.columns, columns=['Coefficient'])
print(coefficients)


df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()

print("Correlation Matrix:")
print(corr['Salary'].sort_values(ascending=False))

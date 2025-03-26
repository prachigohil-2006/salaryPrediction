import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open('./models/linear_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


df = pd.read_csv('./data/Salary_Data.csv')
df = df.dropna()


X = df[['Years of Experience', 'Education Level', 'Job Title']]
y = df['Salary']

with open('./models/one_hot_categories.pkl', 'rb') as f:
    one_hot_categories = pickle.load(f)


X = pd.get_dummies(X, columns=['Education Level', 'Job Title'], drop_first=True)
for category in one_hot_categories:
    if category not in X.columns:
        X[category] = 0
X = X.reindex(columns=one_hot_categories, fill_value=0)


try:
    expected_columns = model.feature_names_in_
except AttributeError:
    expected_columns = one_hot_categories  


for col in expected_columns:
    if col not in X.columns:
        X[col] = 0


X = X.reindex(columns=expected_columns, fill_value=0)

X = X.loc[:, expected_columns]

X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='skyblue', edgecolors='black', alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.title("actual vs. predicted salary")
    plt.xlabel("actual salary")
    plt.ylabel("predicted salary")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./visuals/actual_vs_predicted.png')  
    plt.show()

plot_actual_vs_predicted(y, y_pred)




import pickle
import pandas as pd

dataset_path = './data/Salary_Data.csv'
df = pd.read_csv(dataset_path)

valid_education = df['Education Level'].unique().tolist()
valid_job_titles = df['Job Title'].unique().tolist()

model_path = './models/linear_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

print("\nSALARY PREDICTION MODEL!!\n")

while True:
    try:
        experience = float(input("enter the no.of years of experience: "))
        break
    except ValueError:
        print("invalid input")

while True:
    education = input(f"enter education level {valid_education}: ").strip()
    if education in valid_education:
        break
    print(f"invalid input!please choose from: {valid_education}")

while True:
    job_title = input(f"enter job title {valid_job_titles}: ").strip()
    if job_title in valid_job_titles:
        break
    print(f"invalid input!please choose from: {valid_job_titles}")

input_data = pd.DataFrame({
    'Years of Experience': [experience],
    'Education Level': [education],
    'Job Title': [job_title]
})

# One-Hot Encode input data (must match training format)
#converts to numeric type 
input_data = pd.get_dummies(input_data, columns=['Education Level', 'Job Title'], drop_first=True)


expected_columns = model.feature_names_in_  #LinearRegression has specific columns it was trained on.This line ensures we match those columns when giving new input.

# Add missing columns with zeros for compatibility
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match model input
input_data = input_data.reindex(columns=expected_columns)

predicted_salary = model.predict(input_data)[0]
print(f"\nPREDICTED SALARY: {predicted_salary:.2f}")

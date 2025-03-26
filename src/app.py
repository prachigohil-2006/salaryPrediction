import gradio as gr
import pickle
import pandas as pd
import matplotlib.pyplot as plt

with open('../models/linear_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('../models/one_hot_categories.pkl', 'rb') as f:
    expected_columns = pickle.load(f)


df = pd.read_csv('../data/Salary_Data.csv')
df = df.dropna()

def predict_salary(experience, education, job_title):
    try:
        
        input_data = pd.DataFrame({
            'Years of Experience': [experience],
            'Education Level': [education],
            'Job Title': [job_title]
        })

        # One-hot encode
        input_data = pd.get_dummies(input_data, columns=['Education Level', 'Job Title'])

       
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        input_data[['Years of Experience']] = scaler.transform(input_data[['Years of Experience']])

       
        predicted_salary = model.predict(input_data)[0]
        return f"Predicted Salary ==> {predicted_salary:.2f}"
    except Exception as e:
        return f"Error: {e}"


def plot_actual_vs_predicted():
    return "../visuals/actual_vs_predicted.png"


# def plot_feature_importance():
#     return "../visuals/feature_importance.png"


with gr.Blocks() as demo:
    gr.Markdown("<h1>SALARY PREDICTION WEB-APP👻<h1>")
    gr.Markdown("[this model has 0.86 r2 score]")
    
 
    experience = gr.Number(label="Years of Experience", value=1)
    education = gr.Dropdown(list(df['Education Level'].unique()), label="Education Level")
    job_title = gr.Dropdown(list(df['Job Title'].unique()), label="Job Title")
    
  
    output_text = gr.Textbox(label="Prediction")
    output_image1 = gr.Image(label="Actual vs Predicted Plot")
   
    predict_button = gr.Button("Predict Salary")
    visualize_button1 = gr.Button("View Actual vs Predicted")
   
    
    # Actions
    predict_button.click(fn=predict_salary,
                          inputs=[experience, education, job_title],
                          outputs=output_text)
    
    visualize_button1.click(fn=plot_actual_vs_predicted,
                             inputs=[],
                             outputs=output_image1)
    

demo.launch(allowed_paths=["C:\\Users\\Prachi Gohil\\Downloads\\salary_prediction\\visuals"])

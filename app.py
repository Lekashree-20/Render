from flask import Flask, request, jsonify
import pickle
import pandas as pd
import google.generativeai as genai
from flask_cors import CORS
#from flask_ngrok import run_with_ngrok

# Configure the Gemma API
genai.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the model and LabelEncoder from the pickle files
with open('kidney_disease_voting_model.pkl', 'rb') as model_file:
    loaded_voting_clf = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    loaded_lb = pickle.load(le_file)

# Initialize the Flask application
app = Flask(__name__)
CORS(app)
#run_with_ngrok(app)

# Function to generate a prevention report
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        - Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        - Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        - Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.
    
    2. **Risk Description**
        - Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        - Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        - Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).
    
    3. **Stage of Risk**
        - Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        - Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.
    
    4. **Risk Assessment**
        - Impact on Health: Explore how this specific risk factor might affect various aspects of health (e.g., cardiovascular, metabolic, etc.).
        - Modifiable vs. Non-Modifiable Risks: Distinguish between risks that can be changed (e.g., lifestyle factors) and those that cannot (e.g., genetic predisposition).
        - Comparative Risk: Compare the individual's risk to average levels in the general population or among peers.
        
    5. **Findings**
        - In-Depth Health Observations: Summarize the key findings from the assessment, explaining any critical areas of concern.
        - Diagnostic Insights: Provide insights into how the disease was identified, including the symptoms, biomarkers, or other diagnostic criteria used.
        - Data Interpretation: Offer a more detailed interpretation of the user's health data, explaining what specific values or results indicate.
    
    6. **Recommendations**
        - Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        - Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        - Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.
        
    7. **Way Forward**
        - Next Steps: Provide a clear path forward, including short-term and long-term goals for managing the identified risk or disease.
        - Preventive Measures: Highlight preventive strategies to avoid worsening the condition or preventing its recurrence.
        - Health Resources: Suggest additional resources, such as apps, websites, or support groups, that could help the individual manage their health.
        
    8. **Conclusion**
        - Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        - Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.
    
    9. **Contact Information**
        - Professional Guidance: Include information on how to get in touch with healthcare providers for more personalized advice or follow-up.
        - Support Services: List any available support services, such as nutritionists, fitness coaches, or mental health professionals, that could assist in managing the risk.
    
    10. **References**
        - Scientific Sources: Provide references to the scientific literature or authoritative health guidelines that support the information and recommendations given in the report.
        - Further Reading: Suggest articles, books, or other educational materials for the individual to learn more about their condition and how to manage it.

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """
    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# API endpoint to get user input and generate predictions and reports
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract user input values
        user_input = [float(data.get(col, 0)) for col in data['features']]
        age = data.get('age', 0)

        # Convert the input to a DataFrame
        sample_input = pd.DataFrame([user_input], columns=data['features'])

        # Predict probabilities
        probabilities = loaded_voting_clf.predict_proba(sample_input)

        # Determine the classes and their corresponding risk factors
        class_labels = loaded_lb.inverse_transform([0, 1, 2])  # Convert the encoded classes back to their original labels
        predicted_class_index = probabilities[0].argmax()
        predicted_class = class_labels[predicted_class_index]
        risk_score = probabilities[0][predicted_class_index]

        # Generate the prevention report
        report = generate_prevention_report(
            risk=risk_score, 
            disease=predicted_class, 
            age=age
        )
        response = {
            "generated_report": report,
            "predicted_class": predicted_class,
            "risk_score": risk_score
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
    #app.run(debug=True)


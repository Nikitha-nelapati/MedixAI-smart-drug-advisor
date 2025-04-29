import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF # type: ignore
import pandas as pd
# Load the trained model and TF-IDF vectorizer
model = joblib.load('drug_symptom_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Streamlit page configuration
st.set_page_config(page_title="AI Drug Recommendation", page_icon="💊", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    h1 {
        text-align: center;
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('💊 AI Drug Recommendation System')

# User input for symptoms
symptoms_input = st.text_area('Enter Symptoms (comma separated)', '')

# Prediction and other actions
if st.button('Recommend Drug'):
    if symptoms_input:
        # Convert symptoms to TF-IDF vector
        symptoms_vector = tfidf.transform([symptoms_input])

        # Predict the drug
        predicted_drug = model.predict(symptoms_vector)

        st.success(f'🏥 Recommended Drug: {predicted_drug[0]}')

        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Patient Drug Recommendation Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(100, 10, txt=f"Symptoms: {symptoms_input}")
        pdf.ln(5)
        pdf.cell(100, 10, txt=f"Recommended Drug: {predicted_drug[0]}")
        pdf_output = "patient_report.pdf"
        pdf.output(pdf_output)

        # Download PDF
        with open(pdf_output, "rb") as file:
            btn = st.download_button(
                label="Download Recommendation Report (PDF)",
                data=file,
                file_name="patient_report.pdf",
                mime="application/octet-stream"
            )

        st.info("✅ Patient data saved to the dataset (drug_with_symptoms.csv)")

        # Save the new data to CSV
        new_data = {
            'Age': 25,  # You can add more fields
            'Sex': 'M',  # Example
            'BP': 'NORMAL',  # Example
            'Cholesterol': 'NORMAL',  # Example
            'Na_to_K': 12.0,  # Example
            'Symptoms': symptoms_input,
            'Drug': predicted_drug[0]
        }
        if isinstance(new_data, dict):
           new_data = pd.DataFrame([new_data])  # Convert the dictionary to DataFrame
        print(type(new_data))

        df_existing = pd.read_csv('drug_with_symptoms.csv')
        df_existing = pd.concat([df_existing, new_data, new_data], ignore_index=True)

        df_existing.to_csv('drug_with_symptoms.csv', index=False)

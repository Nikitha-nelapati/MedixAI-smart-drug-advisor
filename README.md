# MedixAI-smart-drug-advisor

This project is a **machine learning-powered web application** built with **Streamlit** that recommends drugs based on user-entered symptoms. It utilizes a **Multinomial Naive Bayes** classifier trained on a dataset of symptoms and corresponding drugs.

The system allows users to:
- Input symptoms in plain text.
- Receive a predicted drug recommendation.
- Generate a **downloadable PDF report**.
- Save new user data to the dataset (`drug_with_symptoms.csv`).

## Technologies Used

- Python 3.11
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- FPDF (for PDF generation)
  
## Project folder structure
drug_recommendation_system/
│
├── app.py                      # Streamlit UI for prediction
├── model.py                     # Trains and saves ML model
├── drug_with_symptoms.csv        # Dataset with symptoms and drug mappings
├── drug_symptom_model.pkl        # Saved trained model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── patient_report.pdf            # Output PDF (temporary)

## Prepare the Dataset
drug_with_symptoms.csv

## Install Required Libraries
pip install pandas scikit-learn streamlit joblib fpdf
python model.py                  #to train the model This will: Preprocess the data, Train a Multinomial Naive Bayes classifier and  Save: drug_symptom_model.pkl, tfidf_vectorizer.pkl
streamlit run app.py #run the application Your browser will open at http://localhost:XXXX

## sample input and output:
Symptoms: headache, fever, nausea
Recommended Drug: Paracetamol

## Description of Key Files
app.py	Streamlit UI with model loading, prediction, PDF generation
model.py	Data cleaning, TF-IDF transformation, model training, and saving
drug_with_symptoms.csv	Dataset with historical records
drug_symptom_model.pkl	Trained ML model (Naive Bayes)
tfidf_vectorizer.pkl	Fitted TF-IDF vectorizer for symptoms
patient_report.pdf	PDF generated after each prediction





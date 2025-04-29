import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv('drug_with_symptoms.csv')

# Handle missing data in the 'Symptoms' column
df['Symptoms'] = df['Symptoms'].fillna("No symptoms provided")

# Handle missing data in the 'Drug' column (Remove rows with NaN in the 'Drug' column)
df = df.dropna(subset=['Drug'])  # Remove rows where 'Drug' is NaN

# OR - alternatively, you can fill missing values with a placeholder
# df['Drug'] = df['Drug'].fillna("Unknown Drug")

# Preprocessing - Convert symptoms into a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Symptoms'])  # Symptoms are the features
y = df['Drug']  # Drug is the target

# Check if there are any NaN values left in the target
print(f"Missing values in 'Drug' column after cleaning: {y.isna().sum()}")  # This should print 0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, 'drug_symptom_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model trained and saved!")

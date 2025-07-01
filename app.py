import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def predict_message(message, model_path='models/sms_classifier_model.pkl', vectorizer_path='models/sms_vectorizer.pkl'):
    """
    Predict if a message is spam or ham using the saved model and vectorizer.
    
    Args:
        message (str): The message to classify
        model_path (str): Path to the saved model file
        vectorizer_path (str): Path to the saved vectorizer file
    
    Returns:
        tuple: (prediction, probability_scores)
    """
    try:
        # Load the saved model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess and vectorize the message
        message_vec = vectorizer.transform([message.lower()])
        
        # Make prediction
        prediction = model.predict(message_vec)[0]
        probabilities = model.predict_proba(message_vec)[0]
        
        return prediction, {'ham': probabilities[0], 'spam': probabilities[1]}
    
    except FileNotFoundError:
        print("Error: Model or vectorizer file not found. Please train the model first.")
        return None, None
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None

# Data Preparation
print("Loading and preparing the dataset...")
print("-" * 40)

# Load and clean the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Simple normalization
df['text'] = df['text'].str.lower()

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:")
print(df['label'].value_counts())
print(f"\nSample messages:")
print(df.head(3))

# Splitting and Vectorization
print("\n" + "="*50)
print("Splitting data and converting to numeric features...")
print("-" * 50)

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], 
    test_size=0.3, 
    random_state=42,
    stratify=df['label']  # Ensure balanced split
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Convert text to numeric features using TfidfVectorizer with improved parameters
vectorizer = TfidfVectorizer(
    max_features=5000,          # Limit vocabulary size to top 5000 words
    stop_words='english',       # Remove common English stop words
    ngram_range=(1, 2),         # Use both unigrams and bigrams
    lowercase=True,             # Convert to lowercase
    strip_accents='ascii'       # Remove accents
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature matrix shape (train): {X_train_vec.shape}")
print(f"Feature matrix shape (test): {X_test_vec.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Model Training
print("\n" + "="*50)
print("Training Multinomial Naive Bayes model...")
print("-" * 50)

# Train Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

print("Model training completed!")

# Model Testing and Evaluation
print("\n" + "="*50)
print("Testing model and evaluating performance...")
print("-" * 50)

# Make predictions on entire test set
y_pred = nb_model.predict(X_test_vec)

# Calculate overall accuracy on entire test set
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Total test samples: {len(y_test)}")
print(f"Correctly classified: {sum(y_test == y_pred)}")
print(f"Incorrectly classified: {sum(y_test != y_pred)}")

# Display confusion matrix
print("\nConfusion Matrix:")
print("-" * 20)
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted")
print(f"            Ham   Spam")
print(f"Actual Ham  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"     Spam   {cm[1,0]:4d}  {cm[1,1]:4d}")

# Calculate metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"True Negatives (Ham correctly classified):  {tn}")
print(f"False Positives (Ham classified as Spam):   {fp}")
print(f"False Negatives (Spam classified as Ham):  {fn}")
print(f"True Positives (Spam correctly classified): {tp}")

# Display detailed classification report
print("\nDetailed Classification Report:")
print("-" * 35)
print(classification_report(y_test, y_pred))


print("\n" + "="*50)
print("Saving model and vectorizer...")
print("-" * 50)

# Create models directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created {models_dir} directory")

# Save the trained model and vectorizer using joblib
model_path = os.path.join(models_dir, 'sms_classifier_model.pkl')
vectorizer_path = os.path.join(models_dir, 'sms_vectorizer.pkl')

joblib.dump(nb_model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

# Verify the saved files
print(f"\nSaved model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
print(f"Saved vectorizer file size: {os.path.getsize(vectorizer_path) / 1024:.2f} KB")

print("\n" + "="*50)
print("Testing model loading...")
print("-" * 50)

# Test loading the saved model and vectorizer
try:
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    # Test with a sample message
    test_message = ["Free entry to win a cash prize! Call now!"]
    test_vec = loaded_vectorizer.transform(test_message)
    prediction = loaded_model.predict(test_vec)
    prediction_proba = loaded_model.predict_proba(test_vec)
    
    print("Model loading successful!")
    print(f"Test message: '{test_message[0]}'")
    print(f"Prediction: {prediction[0]}")
    print(f"Prediction probabilities: Ham={prediction_proba[0][0]:.4f}, Spam={prediction_proba[0][1]:.4f}")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")

print("\n" + "="*50)
print("Analysis Complete!")
print("-" * 50)

print("\n" + "="*50)
print("Example usage of prediction function:")
print("-" * 50)

# Example usage of the prediction function
sample_messages = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be suspended. Call this number immediately!",
    "Thanks for the meeting today. I'll send you the documents later."
]

print("Testing sample messages:")
for i, msg in enumerate(sample_messages, 1):
    pred, probs = predict_message(msg)
    if pred is not None:
        print(f"\n{i}. Message: '{msg[:50]}{'...' if len(msg) > 50 else ''}'")
        print(f"   Prediction: {pred.upper()}")
        print(f"   Confidence: Ham={probs['ham']:.3f}, Spam={probs['spam']:.3f}")

print(f"\n{'='*50}")
print("SMS Classifier training and evaluation completed successfully!")
print("Models saved and ready for use.")
print(f"{'='*50}")

# Streamlit Frontend (Web Only, No CLI)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

if __name__ == "__main__":
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Please install it with 'pip install streamlit'.")
    else:
        st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")
        st.title("📱 SMS Spam Classifier")
        st.write("Enter an SMS message below to check if it's spam or ham.")
        user_message = st.text_area("Message", "", height=100)
        classify_btn = st.button("Classify")
        if classify_btn or user_message:
            if user_message.strip() == "":
                st.warning("Please enter a message.")
            else:
                prediction, probabilities = predict_message(user_message)
                if prediction is not None:
                    st.markdown(f"**Prediction:** {'🚫 <span style=\"color:red\"><b>SPAM</b></span>' if prediction == 'spam' else '📧 <span style=\"color:green\"><b>HAM</b></span>'}", unsafe_allow_html=True)
                    st.progress(probabilities['spam'] if prediction == 'spam' else probabilities['ham'])
                    st.write(f"Confidence:")
                    st.write(f"- Ham: {probabilities['ham']*100:.2f}%")
                    st.write(f"- Spam: {probabilities['spam']*100:.2f}%")
                    if prediction == 'spam' and probabilities['spam'] > 0.7:
                        st.error("⚠️ This message appears to be SPAM with high confidence!")
                else:
                    st.error("Could not classify the message. Please try again.")
        st.info("This app uses a Naive Bayes model trained on the classic SMS Spam Collection dataset.")
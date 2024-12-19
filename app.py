import streamlit as st
import PyPDF2
import docx
import spacy
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from roles_skills import categories_skills
from spacy.cli import download

# Download NLTK resources (only needs to be done once)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Try to load the model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    # Download the model if not available
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

    
# Load the necessary models and encoders
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('word2vec_model.pkl', 'rb') as file:
    word2vec_model = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Function to read the PDF file and extract text
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to read the DOCX file and extract text
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to clean the extracted text
def clean_text(text):
    # Step 1: Remove non-alphanumeric characters (except spaces)
    text = re.sub(r"[^A-Za-z0-9.+ ]", " ", text)
    # Step 2: Replace multiple spaces with a single space
    text = re.sub(r"\s{2,}", " ", text)
    # Step 3: Tokenize into sentences
    sentences = sent_tokenize(text)
    # Step 4: Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Tokenize words
        words = word_tokenize(sentence)
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
        # Join the processed words back into a sentence
        processed_sentences.append(" ".join(words))
    # Step 5: Combine processed sentences back into a single string
    return " ".join(processed_sentences)

# Convert the text data into a single numerical representation using Average Word2Vec
def avg_word2vec(word2vec_model, text):
    # If text is a list of words, join them into a single string
    if isinstance(text, list):
        text = ' '.join(text)
    # Tokenize the entire text
    tokens = word_tokenize(text)  # Tokenizing the entire resume text
    word_vectors = [word2vec_model.wv[word.lower()] for word in tokens if word.lower() in word2vec_model.wv]
    # If there are any word vectors, compute the mean
    if word_vectors:
        vector = np.mean(word_vectors, axis=0)  # Mean of word vectors
    else:
        vector = np.zeros(word2vec_model.vector_size)  # Zero vector if no words match
    return vector

# Function to extract keywords from job description
def extract_keywords_from_job_desc(job_desc_text):
    # Tokenize the job description and clean the tokens
    doc = nlp(job_desc_text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return set(tokens)

# Function to predict category based on the cleaned resume text
def predict_category(resume_text):
    tokenized_text = word_tokenize(resume_text.lower())  # Tokenize the cleaned resume text
    vector_text = avg_word2vec(word2vec_model, tokenized_text)  # Get vector representation
    category = rf_model.predict(vector_text.reshape(1, -1))  # Model prediction
    predicted_category = label_encoder.inverse_transform(category)  # Decode predicted category
    return predicted_category[0]

# Function to extract missing skills from the resume based on the predicted role
def extract_missing_skills_from_resume(resume_text, role):
    doc = nlp(resume_text)
    tokens = [token.text.lower() for token in doc]
    # Get the relevant skills for the predicted role
    role_skills_set = set(categories_skills.get(role, []))
    # Find missing skills by checking the difference between the role skills and tokens in the resume
    missing_skills = role_skills_set.difference(tokens)
    return ', '.join(list(missing_skills))

# Function to compare the resume with a job description
def compare_with_job_desc(resume_text, job_desc_text):
    resume_vector = avg_word2vec(word2vec_model, resume_text)
    job_desc_vector = avg_word2vec(word2vec_model, job_desc_text)
    similarity = cosine_similarity([resume_vector], [job_desc_vector])
    return similarity[0][0]

# Function to extract missing keywords from the resume based on job description
def extract_missing_keywords_from_resume(resume_text, job_desc_text):
    # Extract keywords from the job description
    job_desc_keywords = extract_keywords_from_job_desc(job_desc_text)
    # Tokenize the resume text and clean the tokens
    doc = nlp(resume_text)
    resume_tokens = set([token.text.lower() for token in doc if not token.is_stop and not token.is_punct])
    # Find missing keywords by checking the difference between the job description keywords and resume tokens
    missing_keywords = job_desc_keywords.difference(resume_tokens)
    return ', '.join(list(missing_keywords))

# Main function to process the resume and recommend missing skills
def process_resume(file_path, job_desc_text=""):
    file_name = file_path.name
    # Extract text based on file type
    if file_name.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_path)
    elif file_name.endswith(".docx"):
        resume_text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    # Clean the extracted text
    cleaned_resume_text = clean_text(resume_text)
    
    # Predict the category of the resume
    predicted_category = predict_category(cleaned_resume_text)
    
    # Recommend missing skills
    missing_skills = extract_missing_skills_from_resume(cleaned_resume_text, predicted_category)
    
    # Compare job description keywords with the resume to find missing keywords
    missing_keywords = extract_missing_keywords_from_resume(cleaned_resume_text, job_desc_text) if job_desc_text else None
    
    # Compare the resume with the job description for a match score
    job_match_score = compare_with_job_desc(cleaned_resume_text, job_desc_text) if job_desc_text else None
    
    return predicted_category, missing_skills, missing_keywords, job_match_score


# Custom CSS for styling
st.markdown("""
    <style>
        /* Customizing Sidebar */
        .css-1d391kg {
            background-color: #f7f7f7;
            color: #000000;
        }
        .css-1d391kg a {
            color: #0000ff !important;
        }
        .css-1d391kg h1 {
            color: #1f1f1f;
        }
        .css-1d391kg ul {
            list-style-type: none;
            padding-left: 0;
        }
        .css-1d391kg li {
            margin: 10px 0;
        }
        
        /* Customizing the Title Text */
        h1, h2 {
            color: #2e6df7;
        }
        h3 {
            color: #2e6df7;
        }

        /* Customizing button styles */
        .css-1r6slb5 {
            background-color: #4CAF50;
            color: white;
        }

        /* Customizing the container background */
        .stApp {
            background-color: #f4f7fc;
        }

        /* Customizing warning message */
        .stWarning {
            background-color: #ffcc00;
            color: black;
        }

    </style>
""", unsafe_allow_html=True)

# Main function for Streamlit UI
def main():
    # Sidebar
    st.sidebar.title("Resume Skill Recommendation System")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Resume Predictor", "Job Description Comparison"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ## Features:
        - **Resume Job Role Prediction**: Upload your resume to predict the most suitable job role.
        - **Job Description Comparison**: Paste a job description and compare it with your resume.
    """)
    
    
    # Home Page
    if page == "Home":
        st.title("👋 Welcome to the Resume Skill Recommender App!")
        st.image("https://miro.medium.com/v2/resize:fit:2000/1*ET2aiKwnLubVd_tUNsY8Eg.jpeg", use_container_width=True)
        st.markdown("""
            This app helps you take the next step in your career by:
            - **Predicting the best job role** based on your resume 🧑‍💼💼.
            - **Comparing your resume with a job description** to find missing skills or keywords 📝📊.
        
            Whether you're a fresh graduate or an experienced professional, this app can help you tailor your resume to the job you're aiming for.

            ### Features:
            - **Resume Job Role Prediction**: Upload your resume, and we'll predict the best-fit job role for you based on your skills and experience.
            - **Job Description Comparison**: Paste your job description and upload your resume to compare the two and see if your resume aligns with the job’s requirements.
        
            #### Why Use This App? 
            - **Smart Predictions**: The app analyzes your resume using machine learning to recommend the right job role and identify missing skills.
            - **Optimize Your Resume**: Compare your resume with any job description to ensure you're highlighting the right skills.
            - **Improve Your Career Prospects**: By understanding what employers are looking for and aligning your resume accordingly, you can increase your chances of landing the job you want! 🚀

            ### How it Works:
            1. **Upload Your Resume**: Choose a resume in PDF or DOCX format.
            2. **Predict the Role**: Get suggestions for suitable job titles and missing skills.
            3. **Compare with Job Description**: Copy and paste a job description to check your resume’s match.
            
            📝 **Start by choosing a page from the sidebar!** 📝
        """)

    # Resume Predictor Page
    elif page == "Resume Predictor":
        st.title("🔮 Resume Job Role Prediction")
        st.image("https://miro.medium.com/v2/resize:fit:1200/0*upQaFSWPvVAag9Q-.jpeg", use_container_width=True)
        
        # Add a description to the sidebar
        st.sidebar.title("How to Use Resume Job Role Predictor")
        st.sidebar.markdown("""
            1. **Upload Your Resume**: Upload your resume in PDF or DOCX format.
            2. **Get Role Prediction**: Based on your resume content, we will predict the most suitable job role for you.
            3. **View Missing Skills**: We also suggest keywords and skills you can add to your resume to improve your chances.
            
            📈 **Tip**: Make sure to match your resume with job descriptions to tailor it effectively and increase your chances of getting shortlisted! 🎯
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])
        
        # Process and display the result if file is uploaded
        if uploaded_file is not None:
            predicted_category, missing_skills, _, _ = process_resume(uploaded_file)
            st.write(f"**Your Resume is suitable for this Job Role**: {predicted_category} 🎯")
            st.write(f"**Add these keywords to stand out**: {missing_skills} 🚀")

    # Job Description Comparison Page
    elif page == "Job Description Comparison":
        st.title("🔍 Compare Resume with Job Description")
        st.image("https://www.talentprise.com/wp-content/uploads/2024/08/job-matching.png", use_container_width=True)
        
        # Add a description to the sidebar
        st.sidebar.title("How to Use")
        st.sidebar.markdown("""
            1. **Upload Resume**: Upload your resume in PDF or DOCX format.
            2. **Paste Job Description**: Paste the job description text in the provided box.
            3. **Submit for Comparison**: Click the "Submit Comparison" button to get results.
            4. **Match Score and Feedback**: We provide a match score and recommendations for improvement.
            
            📈 **Tip**: A score above **90%** increases your chances of getting shortlisted. Focus on matching key skills and keywords from the job description!
        """)
        
        # File uploader and job description text box
        uploaded_file = st.file_uploader("Upload a Resume (PDF or DOCX)", type=["pdf", "docx"])
        job_desc_text = st.text_area("Paste Job Description Here", height=250)
        
        # Check if both resume and job description are uploaded and clicked the submit button
        if st.button("Submit Comparison"):
            if uploaded_file is None:
                st.warning("🚨 Please upload a resume first.", icon="⚠️")
            elif not job_desc_text:
                st.warning("🚨 Please paste a job description for comparison.", icon="⚠️")
            else:
                # Process the resume and job description if both are provided
                _, _, missing_keywords, job_match_score = process_resume(uploaded_file, job_desc_text)
                
                # Display the match score and missing keywords
                job_match_score *= 100
                st.write(f"**Resume vs Job Description Match Score**: {job_match_score:.2f} 📊")
                st.write(f"**Add these words to increase the score**: {missing_keywords} ⚠️")
                
                # Categorize the match score and give feedback
                if job_match_score >= 90:
                    st.success("🌟 Excellent Match! You're on the right track!")
                elif job_match_score >= 75:
                    st.warning("👍 Good Match! You’re close. Slight improvements can make it outstanding!")
                elif job_match_score >= 50:
                    st.info("⚠️ Fair Match. Consider adding more relevant skills and keywords.")
                else:
                    st.error("🚨 Bad Match. Strong improvements needed to better align your resume with the job description.")
                
                # Advise for a score above 90 for better chances of shortlisting
                st.markdown("""
                    **💡 Tip**: Try to aim for a match score of **90+** for better chances of getting shortlisted for an interview. 🏆
                """)

    # Footer section
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; font-size: 12px; color: #808080;">
            This is an ML model, can't be accurate all the time, Kindly understand. 🙏
        </div>
    """, unsafe_allow_html=True)

# Run the main function to launch the app
if __name__ == "__main__":
    main()

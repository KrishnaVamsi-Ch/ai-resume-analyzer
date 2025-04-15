import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

def calculate_match(resume_text, job_text):
    texts = [resume_text, job_text]
    tfidf = TfidfVectorizer().fit_transform(texts)
    match_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(match_score * 100, 2)

if __name__ == "__main__":
    resume_path = "sample_resume.pdf"
    job_path = "job_description.txt"

    resume_raw = extract_text_from_pdf(resume_path)
    job_raw = open(job_path, "r", encoding="utf-8").read()

    resume_clean = clean_text(resume_raw)
    job_clean = clean_text(job_raw)

    score = calculate_match(resume_clean, job_clean)
    print(f"📊 Resume Match Score: {score}%")

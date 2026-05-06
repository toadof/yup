5
import random
import gensim.downloader as api

# Load a pre-trained word embedding model
model = api.load("glove-wiki-gigaword-50")  # 50D GloVe embeddings

def get_similar_words(seed_word, top_n=5):
    """Retrieve similar words for the given seed word."""
    try:
        similar_words = [word for word, _ in model.most_similar(seed_word, topn=top_n)]
        return similar_words
    except KeyError:
        return []

def create_paragraph(seed_word):
    """Generate a short paragraph using the seed word and its similar words."""
    similar_words = get_similar_words(seed_word)

    if not similar_words:
        return f"Could not find similar words for '{seed_word}'. Try another word!"

    # Create a simple paragraph
    paragraph = (
        f"Once upon a time, a {seed_word} embarked on a journey. Along the way, it encountered "
        f"a {random.choice(similar_words)}, which led it to a hidden {random.choice(similar_words)}. "
        f"Despite the challenges, it found {random.choice(similar_words)} and embraced the "
        f"adventure with {random.choice(similar_words)}. In the end, the journey was a tale of "
        f"{random.choice(similar_words)} and discovery."
    )

    return paragraph

# Example usage
seed_word = input("Enter a seed word: ").strip().lower()
print("\nGenerated Story:\n")
print(create_paragraph(seed_word))

6
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of the input text using Hugging Face pipeline."""
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

# Example usage
while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        break
    print(analyze_sentiment(user_input))

7
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Take user input for the text passage
text = input("Enter the text you want to summarize:\n")

# Summarize the text
summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

# Print the summarized text
print("\nSummarized Text:")
print(summary[0]['summary_text'])

8
from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere

# Set your Cohere API key
COHERE_API_KEY = "YOUR_COHERE_API"  # Replace with your actual API key

# Read the text file
file_path = "Artificial_Intelligence.txt"  # Replace with your file name

with open(file_path, "r", encoding="utf-8") as file:
    document_text = file.read()

print("File loaded successfully!")

# Create a simple prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in a simple way:\n\n{text}"
)

# Initialize the Cohere model
llm = Cohere(cohere_api_key=COHERE_API_KEY)

# Run the text through Cohere
output = llm.invoke(prompt_template.format(text=document_text))

# Display the result
print("Summary:\n", output)

9
from pydantic import BaseModel
import wikipediaapi

# Define the Pydantic schema
class InstitutionDetails(BaseModel):
    name: str
    founder: str
    founded_year: str
    branches: str
    employees: str
    summary: str

# Wikipedia extraction function
def fetch_institution_details(institution_name: str) -> InstitutionDetails:
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent="MyWikipediaScraper/1.0 (contact: myemail@example.com)",
        language="en"
    )
    page = wiki_wiki.page(institution_name)

    if not page.exists():
        raise ValueError("Institution page does not exist on Wikipedia")

    # Extract information (this part needs actual content parsing)
    summary = " ".join(page.summary.split(".")[:4]) + "."

    # Placeholder extraction logic
    founder = "Not Available"
    founded_year = "Not Available"
    branches = "Not Available"
    employees = "Not Available"

    for section in page.sections:
        if "founder" in section.title.lower():
            founder = section.text.split(". ")[0]
        if "founded" in section.title.lower():
            founded_year = section.text.split(". ")[0]
        if "branches" in section.title.lower():
            branches = section.text.split(". ")[0]
        if "employees" in section.title.lower():
            employees = section.text.split(". ")[0]

    return InstitutionDetails(
        name=institution_name,
        founder=founder,
        founded_year=founded_year,
        branches=branches,
        employees=employees,
        summary=summary
    )

# Example invocation
institution_name = input("Enter Institution Name: ")
try:
    details = fetch_institution_details(institution_name)
    print(details.model_dump_json(indent=4))
except ValueError as e:
    print(str(e))


10
import requests
import PyPDF2
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── STEP 1: Download IPC PDF ───────────────────────────────────────────────

def download_ipc_pdf(url, save_path="ipc.pdf"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded IPC PDF to: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# ─── STEP 2: Extract text from PDF ──────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# ─── STEP 3: Preprocess text ────────────────────────────────────────────────

def preprocess_text(text):
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# ─── STEP 4: Create index ───────────────────────────────────────────────────

def create_index(text):
    index = {}
    try:
        section_pattern = r"((?:CHAPTER|SECTION)\s+\w+\.?\s+.*?)(?:(?:CHAPTER|SECTION)\s+\w+\.?\s+|$)"
        matches = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            title_match = re.search(r"^(?:CHAPTER|SECTION)\s+\w+\.?\s+(.*?)(?=\n)", match, re.DOTALL | re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                content = match[title_match.end():].strip()
                index[title] = content
        return index
    except Exception as e:
        print(f"Error creating index: {e}")
        return {}

# ─── STEP 5: Find most relevant section ─────────────────────────────────────

def get_most_relevant_section(query, index):
    try:
        if not index:
            return None
        sections = list(index.values())
        section_titles = list(index.keys())
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sections + [query])
        query_vector = tfidf_matrix[-1]
        similarities = cosine_similarity(query_vector, tfidf_matrix[:-1]).flatten()
        if not similarities.any():
            return None
        most_relevant_index = similarities.argmax()
        return section_titles[most_relevant_index]
    except Exception as e:
        print(f"Error finding relevant section: {e}")
        return None

def get_section_text(section_title, index):
    try:
        return index.get(section_title)
    except Exception as e:
        print(f"Error getting section text: {e}")
        return None

# ─── STEP 6: Generate response ──────────────────────────────────────────────

def generate_response(query, index):
    if not index:
        return "I'm sorry, I cannot process the IPC without the index. Please ensure the IPC content is loaded."
    relevant_section_title = get_most_relevant_section(query, index)
    if not relevant_section_title:
        return "I'm sorry, I couldn't find relevant information in the IPC for your query."
    section_text = get_section_text(relevant_section_title, index)
    if not section_text:
        return "I found the relevant section, but I'm unable to retrieve the details."
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', section_text.strip())
    cleaned_text = re.sub(r'[\t]+\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    response = f"Here's what I found in the Indian Penal Code, **{relevant_section_title}**:\n\n{cleaned_text}"
    return response

# ─── STEP 7: Chatbot loop ────────────────────────────────────────────────────

def chatbot(index):
    print("Welcome to the Indian Penal Code Chatbot! Ask me anything about the IPC. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        response = generate_response(query, index)
        print(f"Chatbot: {response}\n")

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')

    ipc_pdf_url = "https://www.indiacode.nic.in/bitstream/123456789/4219/1/THE-INDIAN-PENAL-CODE-1860.pdf"
    download_ipc_pdf(ipc_pdf_url)

    pdf_path = "ipc.pdf"
    ipc_text = extract_text_from_pdf(pdf_path)
    if ipc_text:
        ipc_index = create_index(ipc_text)
        if ipc_index:
            chatbot(ipc_index)
        else:
            print("Failed to create index. Chatbot cannot start.")
    else:
        print("Failed to extract text from PDF. Chatbot cannot start.")

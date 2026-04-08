PROGRAM 1
Explore pre-trained word vectors. Explore word relationships using vector arithmetic.
Perform arithmetic operations and analyze results.
import gensim.downloader as api
print("Loading model... (This may take a while)")
model = api.load("word2vec-google-news-300")
print("Model loaded!")
def find_similar(word):
 try:
 similar_words = model.most_similar(word)
 print(f"\nWords similar to '{word}':")
 for w, score in similar_words[:5]:
 print(f"{w}: {score:.4f}")
 except KeyError:
 print(f"'{word}' not found in the vocabulary.")
def word_arithmetic(word1, word2, word3):
 try:
 result = model.most_similar(positive=[word1, word2], negative=[word3])
 print(f"\n'{word1}' - '{word3}' + '{word2}' = '{result[0][0]}' (Most similar
word)")
 except KeyError as e:
 print(f"Error: {e}")
def check_similarity(word1, word2):
 try:
 similarity = model.similarity(word1, word2)
 print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
 except KeyError as e:
 print(f"Error: {e}")
def odd_one_out(words):
 try:
 odd = model.doesnt_match(words)
 print(f"\nOdd one out from {words}: {odd}")
 except KeyError as e:
 print(f"Error: {e}")
find_similar("king")
word_arithmetic("king", "woman", "man")
check_similarity("king", "queen")
odd_one_out(["apple", "banana", "grape", "car"])
PROGRAM 2
Use dimensionality reduction (e.g., PCA or t-SNE) to visualize word embeddings. Select 10
words from a specific domain (e.g., sports, technology) and visualize their embeddings.
Analyze clusters and relationships. Generate contextually rich outputs using embeddings.
Write a program to generate 5 semantically similar words for a given input.
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
print("Loading model... (This may take a while)")
model = api.load("word2vec-google-news-300")
print("Model loaded!")
tech_words = ["computer", "algorithm", "software", "hardware", "AI",
 "cloud", "database", "network", "cybersecurity", "encryption"]
word_vectors = np.array([model[word] for word in tech_words])
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)
plt.figure(figsize=(8,6))
for word, (x, y) in zip(tech_words, reduced_vectors):
 plt.scatter(x, y)
 plt.text(x+0.02, y+0.02, word, fontsize=12)
plt.title("2D Visualization of Technology Word Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()
def find_similar_words(word):
 try:
 similar_words = model.most_similar(word, topn=5)
 print(f"\n5 words similar to '{word}':")
 for w, score in similar_words:
 print(f"{w}: {score:.4f}")
 except KeyError:
 print(f"'{word}' not found in the vocabulary.")
find_similar_words("AI")
PROGRAM 3
Train a custom Word2Vec model on a small dataset. Train embeddings on a domain-specific
corpus (e.g., legal, medical) and analyze how embeddings capture domain-specific
semantics.
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
medical_corpus = [
 "The doctor diagnosed the patient with diabetes.",
 "Insulin is used to treat diabetes.",
 "A cardiologist specializes in heart diseases.",
 "Patients with hypertension should reduce salt intake.",
 "Antibiotics treat bacterial infections.",
 "Vaccines help build immunity.",
 "Surgery removes tumors.",
 "A neurologist treats nervous system disorders.",
]
nltk.download('punkt')
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in medical_corpus]
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, min_count=1,
workers=4)
model.save('medical_word2vec.model')
def plot_embeddings(model):
 words = list(model.wv.index_to_key)
 word_vectors = model.wv[words]
 tsne = TSNE(n_components=2, random_state=42)
 reduced_vectors = tsne.fit_transform(word_vectors)
 plt.figure(figsize=(8, 5))
 for i, word in enumerate(words):
 x, y = reduced_vectors[i]
 plt.scatter(x, y)
 plt.annotate(word, (x, y), fontsize=10)
 plt.title("Word Embeddings Visualization")
 plt.show()
plot_embeddings(model)
def find_similar_words(word):
 if word in model.wv:
 similar_words = model.wv.most_similar(word, topn=5)
 print(f"Words similar to '{word}':")
 for similar, score in similar_words:
 print(f"{similar} (Similarity: {score:.2f})")
 else:
 print(f"'{word}' not found in vocabulary")
find_similar_words("diabetes")
PROGRAM 4
Use word embeddings to improve prompts for Generative AI model. Retrieve similar words
using word embeddings. Use the similar words to enrich a GenAI prompt. Use the AI model
to generate responses for the original and enriched prompts. Compare the outputs in terms
of detail and relevance.
import gensim.downloader as api
from transformers import pipeline
embedding_model = api.load("glove-wiki-gigaword-100")
original_prompt = "Describe the beautiful landscapes during sunset."
def enrich_prompt(prompt, embedding_model, n=5):
 words = prompt.split()
 enriched_prompt = []
 for word in words:
 word_lower = word.lower()
 if word_lower in embedding_model:
 similar_words = embedding_model.most_similar(word_lower, topn=n)
 similar_word_list = [w[0] for w in similar_words]
 enriched_prompt.append(" ".join(similar_word_list))
 else:
 enriched_prompt.append(word)
 return " ".join(enriched_prompt)
enriched_prompt = enrich_prompt(original_prompt, embedding_model)
generator = pipeline("text-generation", model="gpt2")
original_response = generator(original_prompt, max_length=50, num_return_sequences=1)
enriched_response = generator(enriched_prompt, max_length=50, num_return_sequences=1)
print("Original prompt response")
print(original_response[0]['generated_text'])
print("\nEnriched prompt response")
print(enriched_response[0]['generated_text'])

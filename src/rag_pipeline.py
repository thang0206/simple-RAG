import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
# TensorFlow and PyTorch can be integrated for advanced LLMs or neural retrievers
# Experience: TensorFlow (3 years), PyTorch (3 years), Scikit-learn (3 years)

class SimpleRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=2):
        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus[i], scores[i]) for i in top_indices]

class SimpleLLM:
    def generate(self, query, retrieved_docs):
        context = " ".join([doc for doc, _ in retrieved_docs])
        return f"Answer based on context: {context}\nQuestion: {query}"

def load_corpus(path="corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def rag_pipeline(query):
    corpus = load_corpus()
    retriever = SimpleRetriever(corpus)
    retrieved = retriever.retrieve(query)
    llm = SimpleLLM()
    answer = llm.generate(query, retrieved)
    return answer

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def authenticate():
    if not API_KEY or API_KEY == "your_openai_api_key_here":
        print("Please set a valid OPENAI_API_KEY in the .env file.")
        exit(1)

if __name__ == "__main__":
    authenticate()
    query = input("Enter your question: ")
    answer = rag_pipeline(query)
    print("\n--- RAG Answer ---\n", answer)

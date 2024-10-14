import os
import math
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Utility functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    return [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]

def term_frequency(term, document):
    return document.count(term) / len(document)

def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (num_docs_containing_term)) if num_docs_containing_term > 0 else 0

def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string if soup.title else ""
    body = soup.body.get_text(separator=" ") if soup.body else ""
    return title, body

def load_html_files(folder_path):
    documents = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                title, body = extract_text_from_html(content)
                documents.append(clean_text(title + " " + body))
                filenames.append(filename)  # Store the actual filename

    return documents, filenames

# Precision@K function
def calculate_precision_at_k(relevant_docs, ranked_docs, k):
    """
    Calculate precision at k for a single query.

    Params:
      relevant_docs...A set of relevant document filenames for the query.
      ranked_docs.....A list of tuples (filename, similarity score).
      k...............The cutoff for precision.

    Returns:
      Precision at k (float).
    """
    if not ranked_docs:
        return 0.0
    top_k_docs = ranked_docs[:k]
    relevant_retrieved = sum(1 for doc in top_k_docs if doc[0] in relevant_docs)  # Check if filename is in relevant_docs
    return relevant_retrieved / k

# Sample queries and relevant docs
queries = [
    "What is Kamala Harris's transformation in the 2024 election?",
    "How are Trump and Harris appealing to undecided voters in 2024?",
    "What were the key moments from the VP debate between Walz and Vance?",
    "Are there concerns about election deniers disrupting the 2024 vote count?",
    "What are the latest 2024 polls saying about Harris and Trump?",
    "How did Harris respond to Trump's stance on abortion in Georgia?",
    "Who is favored to win the 2024 presidential race between Trump and Harris?",
    "What role are celebrities like Taylor Swift playing in the US election?",
    "How does Trump frame his policies differently from Harris in the 2024 election?"
]

# Relevant documents dictionary
relevant_docs_dict = {
    "What is Kamala Harris's transformation in the 2024 election?": {
        "2024 US election_ Kamala Harris's transformation.html",
        "The Harris-Trump debate showed US foreign policy matters in this election _ Chatham House - International Affairs Think Tank.html",
        "Harris Had Stronger Debate, Polls Find, but the Race Remains Deadlocked - The New York Times.html"
    },
    "How are Trump and Harris appealing to undecided voters in 2024?": {
        "Election 2024 Latest_ Trump and Harris campaign for undecided voters with just 6 weeks left _ AP News.html",
        "Who are undecided voters and why_ Here's what 10 of them told NPR _ NPR.html",
    },
    "What were the key moments from the VP debate between Walz and Vance?": {
        "Fact checking VP debate claims from Walz and Vance's 2024 showdown - CBS News.html",
        "VP debate fact check_ Vance and Walz on the economy, immigration and more _ NPR.html"
    },
    "Are there concerns about election deniers disrupting the 2024 vote count?": {
        "Fears mount that election deniers could disrupt vote count in US swing states _ US elections 2024 _ The Guardian.html",
        "`Arm the public with facts'_ Microsoft billionaire fights US election disinformation _ US elections 2024 _ The Guardian.html"
    },
    "What are the latest 2024 polls saying about Harris and Trump?": {
        "Harris and Trump neck-and-neck in polls with early voting under way _ US elections 2024 _ The Guardian.html",
        "US election polls tracker 2024_ Who is ahead - Harris or Trump_.html",
        "Kamala Harris vs. Donald Trump_ Latest Polls in 2024 Presidential Election - The New York Times.html"
    },
    "How did Harris respond to Trump's stance on abortion in Georgia?": {
        "Harris condemns Trump in Georgia after news of abortion-related deaths _ US elections 2024 _ The Guardian.html",
        "Trump bemoans lack of support from Jewish voters and blames `Democrat curse' _ US elections 2024 _ The Guardian.html"
    },
    "Who is favored to win the 2024 presidential race between Trump and Harris?": {
        "Harris vs Trump_ who will win the 2024 presidential election_.html",
        "Kamala Harris vs. Donald Trump_ Latest Polls in 2024 Presidential Election - The New York Times.html"
    },
    "What role are celebrities like Taylor Swift playing in the US election?": {
        "Taylor Swift and Oprah_ do celebrities matter in the US election_.html",
        "Who are undecided voters and why_ Here's what 10 of them told NPR _ NPR.html"
    },
    "How does Trump frame his policies differently from Harris in the 2024 election?": {
        "Trump and Harris vocabularies signal their different frames of mind.html",
        "The Harris-Trump debate showed US foreign policy matters in this election _ Chatham House - International Affairs Think Tank.html",
        "Harris and Trump neck-and-neck in polls with early voting under way _ US elections 2024 _ The Guardian.html"
    }
}

def main():
    folder_path = './dataset/'
    documents, filenames = load_html_files(folder_path)

    # Iterate over each query
    for query in queries:
        # Clean the query
        cleaned_query = clean_text(query)

        # Get unique vocabulary
        vocab = sorted(set(word for doc in documents + [cleaned_query] for word in doc))

        # Compute TF-IDF vectors
        query_vector = compute_tfidf(cleaned_query, documents, vocab)
        doc_vectors = [compute_tfidf(doc, documents, vocab) for doc in documents]

        # Calculate cosine similarity
        similarities = [(filenames[i], cosine_similarity(query_vector, doc_vector)) for i, doc_vector in enumerate(doc_vectors)]

        # Sort similarities
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Get relevant documents for this query
        relevant_docs = relevant_docs_dict.get(query, set())

        # Calculate Precision@K
        k = 5
        precision_at_k = calculate_precision_at_k(relevant_docs, similarities, k)

        # Write results to a file
        with open("evaluation_results.txt", "a") as f:
            f.write(f"Results for query: '{query}'\n")
            f.write(f"Precision@{k}: {precision_at_k:.4f}\n")
            for title, similarity in similarities[:k]:
                f.write(f"Document: {title}, Similarity: {similarity:.4f}\n")
            f.write("\n")

    # Print results after writing to file
    print("Results written to 'evaluation_results.txt'.")
    with open('evaluation_results.txt', 'r') as f:
        content = f.readlines()  # Read lines to preserve the structure
        for line in content:
            # If the line contains "Document:", format the filename
            if line.startswith("Document:"):
                # Extract the document title and similarity score
                parts = line.split(", Similarity:")
                filename = parts[0][10:]  # Remove "Document: " prefix
                similarity_score = parts[1].strip()  # Get the similarity score
                # Format filename
                if len(filename) > 48:
                    formatted_filename = f"{filename[:25]}...{filename[-25:]}, Similarity: {similarity_score}"
                else:
                    formatted_filename = line.strip()  # Keep original if it's short
                print(formatted_filename)
            else:
                print(line.strip())  # Print other lines as is


if __name__ == "__main__":
    main()

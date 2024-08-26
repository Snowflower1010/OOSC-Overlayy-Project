import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel



def scrape_website(url):
    """Scrapes the website for all links."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].startswith('http')]
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []


def fetch_content(links):
    """
    Fetch the content of each link
    """
    content_dict = {}
    for link in links:
        try:
            response = requests.get(link)
            response.raise_for_status()
            content_dict[link] = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content for {link}: {e}")
    return content_dict


def save_to_csv(content_dict, filename):
    """
    Save the content to a CSV file
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Link", "Content"])
        for link, content in content_dict.items():
            writer.writerow([link, content])


def generate_questions(content_dict):
    """
    Generate 10 concise questions for each webpage content using Google Gemini API
    """
    gemini_api_key = "AIzaSyCIMP0fjsiEWNN8N4qwBzs6pFIg-n0_9tM"  
    gemini_url = "https://api.gemini.ai/v1/question-generation"

    questions_dict = {}
    for link, content in content_dict.items():
        payload = {
            "input": content,
            "num_questions": 10,
            "question_type": "concise"
        }
        headers = {
            "Authorization": f"Bearer {gemini_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(gemini_url, headers=headers, json=payload)
            response.raise_for_status()
            questions = response.json().get("questions", [])
            questions_dict[link] = questions
        except requests.exceptions.RequestException as e:
            print(f"Error generating questions for {link}: {e}")
    return questions_dict


def load_questions_and_content(questions_dict):
    """
    Load the questions and content from the questions_dict
    """
    questions = []
    content = []
    for link, questions_list in questions_dict.items():
        questions.extend(questions_list)
        content.extend([questions_dict[link][0]] * len(questions_list))  # Use the first question as the content
    return questions, content


def calculate_similarity(questions, content):
    """
    Calculate the similarity between the questions and content using TF-IDF and cosine similarity.
    Checks are added to ensure non-empty inputs.
    """
    if not questions or not content:
        raise ValueError("Questions or content are empty, cannot calculate similarity.")
    
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    try:
        # Transform questions and content into TF-IDF vectors
        question_vectors = vectorizer.fit_transform(questions)
        content_vectors = vectorizer.transform(content)
    except ValueError as e:
        print(f"Error in vectorization: {e}")
        return None
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(question_vectors, content_vectors)
    
    return similarity_matrix

def select_relevant_links(similarity_matrix, questions_dict):
    """
    Select the 5 most relevant links based on the similarity matrix
    """
    relevant_links = []
    for i, row in enumerate(similarity_matrix):
        link = list(questions_dict.keys())[i]
        scores = row.tolist()
        scores.sort(reverse=True)
        top_scores = scores[:5]
        if all(score > 0.5 for score in top_scores):
            relevant_links.append(link)
    return relevant_links[:5]


def save_final_output(questions_dict, relevant_links, topics):
    """
    Save the generated questions, relevant links, and topics to a JSON file
    """
    data = {
        "topics": topics,
        "relevant_links": relevant_links,
        "questions": {}
    }
    for link, questions_list in questions_dict.items():
        data["questions"][link] = questions_list
    save_to_json(data, "output.json")

def validate_questions(questions_dict):
    """
    Validate that each webpage has 10 questions, each under 80 characters
    """
    for link, questions_list in questions_dict.items():
        if len(questions_list) != 10:
            raise ValueError(f"Link {link} has {len(questions_list)} questions, expected 10")
        for question in questions_list:
            if len(question) > 80:
                raise ValueError(f"Question '{question}' on link {link} is too long (>80 characters)")

def validate_relevant_links(relevant_links):
    """
    Validate that each entry includes 5 relevant links
    """
    if len(relevant_links) != 5:
        raise ValueError(f"Expected 5 relevant links, got {len(relevant_links)}")

def validate_topics(topics):
    """
    Validate that each entry includes topics
    """
    if not topics:
        raise ValueError("Topics are missing")

def save_to_json(questions_dict, relevant_links, topics):
    """
    Save the validated data to a JSON file
    """
    data = {
        "topics": topics,
        "relevant_links": relevant_links,
        "questions": {}
    }
    for link, questions_list in questions_dict.items():
        data["questions"][link] = questions_list
    with open("output.json", "w") as f:
        json.dump(data, f, indent=4)

def main():
    url = input("Enter a website URL: ")
    
    # Step 1: Scrape website for links
    links = scrape_website(url)
    
    # Step 2: Fetch content from the links
    content_dict = fetch_content(links)
    
    # Step 3: Generate questions for each webpage content
    questions_dict = generate_questions(content_dict)
    
    # Step 4: Validate the generated questions
    validate_questions(questions_dict)
    
    # Step 5: Load questions and content for similarity calculation
    questions, content = load_questions_and_content(questions_dict)
    
    # Step 6: Calculate similarity and select relevant links
    similarity_matrix = calculate_similarity(questions, content)
    
    if similarity_matrix is not None:
        relevant_links = select_relevant_links(similarity_matrix, questions_dict)
        
        # Validate the relevant links
        validate_relevant_links(relevant_links)
    
        # Step 7: Generate topics (placeholder, replace with actual topic modeling output)
        documents = [content for content in content_dict.values()]
        dictionary = gensim.corpora.Dictionary(documents)
        bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
        lda_model = LdaModel(bow_corpus, num_topics=3, id2word=dictionary)  # Adjust num_topics as needed
        topics = [lda_model.print_topic(topic_id, topn=5) for topic_id in range(lda_model.num_topics)]
        
        # Validate topics
        validate_topics(topics)
        
        # Step 8: Save the final output to a JSON file
        save_final_output(questions_dict, relevant_links, topics)
        
        print("Processing complete. Output saved to 'output.json'.")
    else:
        print("Processing halted due to an error in similarity calculation.")

def question_quality_score(clarity, relevance, grammar):
    """
    Calculate the Question Quality Score (QQS)
    """
    return (clarity + relevance + grammar) / 3

def question_novelty_score(similarity):
    """
    Calculate the Question Novelty Score (QNS)
    """
    return 1 - similarity

def question_coverage_score(topics_covered, total_topics):
    """
    Calculate the Question Coverage Score (QCS)
    """
    return topics_covered / total_topics

def relevance_score(relevant_links, total_links):
    """
    Calculate the Relevance Score (RS)
    """
    return relevant_links / total_links

def precision(true_positives, false_positives):
    """
    Calculate the Precision
    """
    return true_positives / (true_positives + false_positives)

def recall(true_positives, false_negatives):
    """
    Calculate the Recall
    """
    return true_positives / (true_positives + false_negatives)

def question_relevance_score(qqs, qns, qcs, rs, precision, recall):
    """
    Calculate the Question Relevance Score (QRS)
    """
    return (qqs + qns + qcs) * (rs + precision + recall) / 3

# Example usage
clarity = 4
relevance = 4
grammar = 4
similarity = 0.5
topics_covered = 5
total_topics = 10
relevant_links = 3
total_links = 5
true_positives = 2
false_positives = 1
false_negatives = 2

qqs = question_quality_score(clarity, relevance, grammar)
qns = question_novelty_score(similarity)
qcs = question_coverage_score(topics_covered, total_topics)
rs = relevance_score(relevant_links, total_links)
precision_val = precision(true_positives, false_positives)
recall_val = recall(true_positives, false_negatives)
qrs = question_relevance_score(qqs, qns, qcs, rs, precision_val, recall_val)

print("Question Quality Score (QQS):", qqs)
print("Question Novelty Score (QNS):", qns)
print("Question Coverage Score (QCS):", qcs)
print("Relevance Score (RS):", rs)
print("Precision:", precision_val)
print("Recall:", recall_val)
print("Question Relevance Score (QRS):", qrs)

if __name__ == "__main__":
    main()
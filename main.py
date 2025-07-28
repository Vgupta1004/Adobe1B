# main_1b.py
import fitz
import json
import os
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

def parse_document_sections(pdf_path):
    doc = fitz.open(pdf_path)
    sections = []
    current_text = ""
    current_heading = "Introduction"
    start_page = 1
    for page_num, page in enumerate(doc):
        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
        for b in blocks:
            text = b[4].strip().replace('\n', ' ')
            is_heading = len(text.split()) < 10 and ("bold" in text.lower() or text.isupper())
            if is_heading and current_text:
                sections.append({"title": current_heading, "content": current_text, "page": start_page, "doc_name": os.path.basename(pdf_path)})
                current_heading, current_text, start_page = text, "", page_num + 1
            else:
                current_text += " " + text
    if current_text:
        sections.append({"title": current_heading, "content": current_text, "page": start_page, "doc_name": os.path.basename(pdf_path)})
    doc.close()
    return sections

def main():
    print("Starting Round 1B with TF-IDF approach...")
    with open(os.path.join(INPUT_DIR, 'config.json'), 'r') as f: config = json.load(f)
    persona, job = config['persona'], config['job_to_be_done']

    all_sections, pdf_files = [], [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    for filename in pdf_files:
        print(f"Parsing {filename}...")
        all_sections.extend(parse_document_sections(os.path.join(INPUT_DIR, filename)))

    print("Ranking sections with TF-IDF...")
    query = f"{persona['role']}: {job['task']}"
    
    # Create a corpus of all section content plus the query
    corpus = [section['content'] for section in all_sections]
    corpus.insert(0, query)

    # Vectorize the corpus
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Separate the query vector from the section vectors
    query_vector = tfidf_matrix[0]
    section_vectors = tfidf_matrix[1:]

    # Calculate similarities and rank
    similarities = cosine_similarity(query_vector, section_vectors).flatten()
    for i, section in enumerate(all_sections):
        section['relevance_score'] = similarities[i]

    ranked_sections = sorted(all_sections, key=lambda x: x['relevance_score'], reverse=True)

    # Generate JSON Output
    output_data = {
        "metadata": {"input_documents": pdf_files, "persona": persona, "job_to_be_done": job, "processing_timestamp": datetime.now().isoformat()},
        "Extracted Section": [],
        "Sub-section Analysis": []
    }
    
    for i, section in enumerate(ranked_sections[:10]):
        output_data["Extracted Section"].append({"Document": section['doc_name'], "Page number": section['page'], "Section title": section['title'], "importance_rank": i + 1})
        # For simplicity, using the first few sentences as "Refined Text"
        refined_text = ". ".join(re.split(r'(?<=\.|\?)\s+', section['content'])[:3])
        output_data["Sub-section Analysis"].append({"Document": section['doc_name'], "Refined Text": refined_text, "Page Number": section['page']})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'analysis_output.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Processing complete. Output saved to {os.path.join(OUTPUT_DIR, 'analysis_output.json')}")

if __name__ == '__main__':
    main()
# Adobe1B

# Adobe India Hackathon 2025 - Round 1B Submission

This project is a solution for Round 1B: Persona-Driven Document Intelligence.

---

### Our Approach 

Our solution works by understanding the meaning (semantics) of the documents and the user's request. It follows these steps:

1.  **Document Parsing:** The system first parses all input PDFs to extract clean text content, breaking it down into logical sections based on headings.
2.  **Semantic Embedding:** We use a pre-trained Sentence Transformer model (`all-MiniLM-L6-v2`) to convert the user's request (Persona + Job-to-be-Done) and each document section into numerical vectors (embeddings).
3.  **Relevance Ranking:** The relevance of each document section is calculated by measuring the cosine similarity between its vector and the user's query vector. Sections with the highest similarity are ranked as most important.
4.  **Sub-section Summarization:** For the top-ranked sections, the system performs an extractive summary by finding the most relevant sentences within that section that best match the user's query.

This entire process runs offline within a Docker container and is optimized to be fast and efficient on a CPU.

---

### Models and Libraries Used 

* **Model:** `all-MiniLM-L6-v2` (from the `sentence-transformers` library) for generating text embeddings.
* **Libraries:**
    * `PyMuPDF`: For parsing PDF documents.
    * `sentence-transformers`: For semantic embedding and similarity calculations.
    * `scikit-learn`: For utility functions.

---

### How to Build and Run the Solution 

**1. Prerequisites:**
* Docker must be installed and running.
* Place all input PDFs and a `config.json` file (containing the persona and job) inside an `input` folder in the project root.

**2. Build the Docker Image:**
Navigate to the project's root directory in your terminal and run:
```bash
docker build --platform linux/amd64 -t mysolution1b .
```

**3. Run the Container:**
After the build is complete, run the following command to execute the analysis:
```bash
docker run --rm -v "%cd%/input:/app/input" -v "%cd%/output:/app/output" --network none mysolution1b
```
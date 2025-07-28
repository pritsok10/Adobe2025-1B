# Adobe2025-1B

Overview
This solution automates the extraction and intelligent selection of relevant information from a collection of PDF documents to address user-specified requirements ("persona" and "job to be done"). The approach consists of a two-stage pipeline: (1) outline extraction from PDFs, and (2) semantic analysis and content refinement based on a natural language query. Each stage is modular and scalable, leveraging recent advances in natural language processing for efficient and meaningful information retrieval.

Stage 1: Outline Extraction
Objective:
To quickly index and structure the content of large PDFs by extracting headings and their corresponding locations.

Method:

The script imports a function (process_pdfs) from an external module (goofrun1.py), which processes a directory of PDFs.

Each PDF is parsed to detect prominent headings (likely via font size, type, or embedded bookmarks).

For every heading, metadata such as text, page numbers, and (if available) bounding boxes (bbox) are captured.

Produces a JSON outline file for each PDF, efficiently mapping the document’s structure for downstream analysis.

Benefits:
Saving the document outline enables fast, sectional access in the next stage and avoids repeated costly PDF parsing.

Stage 2: Intelligent Analysis
Objective:
To semantically match user queries with the most relevant sections across all document outlines and extract detailed content from those sections.

Method:

Loads all generated outlines and compiles a list of headings, linking each back to their document and location.

Combines the user's persona and job description into a single query string.

Uses a pre-trained SentenceTransformer (e.g., all-MiniLM-L6-v2) to compute high-dimensional semantic embeddings for both the query and all available headings.

Calculates cosine similarity to identify which headings are most relevant to the query.

Selects the top N (here, 5) highest-matching headings.

For each selected heading, the corresponding PDF is loaded, and all text under that heading—up to the next heading—is extracted programmatically (using precise coordinate bounding if possible).

Aggregates the results in structured JSON for easy downstream consumption or user interface display.

Key Techniques:

Semantic Matching: Sentence embeddings capture semantic similarity, letting the algorithm intelligently link loosely-phrased user needs to document content, even if terminology differs.

Fine-Grained Extraction: Bounding box info ensures the precise retrieval of section text, minimizing irrelevant content.

Design Rationale & Advantages
Separation of concerns: Outline extraction and analysis are decoupled for modularity, improved runtime, and ease of debugging.

Scalable: Can handle many and/or lengthy documents efficiently due to outline-based indexing.

Generalizable: By using sentence embeddings, the method is robust across subject matter, and easily adaptable for other personas/queries.

Conclusion
This two-stage pipeline provides a robust, efficient way to surface relevant information from large document collections, tailored to complex user queries. Its modularity, semantic reasoning, and fine-grained extraction make it highly practical for automating knowledge work.

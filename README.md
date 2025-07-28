# PDF Outline Extraction & Intelligent Analysis

This project extracts structured outlines (titles, headings, hierarchy) from PDF documents and performs intelligent section analysis using semantic search. It is designed for tasks such as travel planning, document summarization, and content extraction.

## Features
- Extracts document outlines (titles, headings, hierarchy) from PDFs using font and layout analysis
- Uses Sentence Transformers for semantic search and section ranking
- Outputs results as structured JSON
- Supports batch processing of multiple PDFs
- Dockerized for easy deployment

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Quick Start

### 1. Clone the repository and place your PDFs
- Place your PDF files in the `Collection/pdf/` directory.

### 2. Build and run with Docker
```sh
cd /path/to/project
# Build the Docker image (replace my1bsol with any name you like)
docker build -t my1bsol .
# Run the container (mounts your project directory for output access)
docker run --rm -v ${PWD}:/app my1bsol
```

### 3. Run locally (without Docker)
```sh
pip install -r requirements.txt
python 1bsol.py
```

### 4. Output
- The main output is `challenge1b_output.json` in the project directory.
- Intermediate and log files are created in `Collection/Outputgoof2/` and removed after completion.

## Project Structure
- `1bsol.py` — Main pipeline script (run this)
- `goofrun1.py` — PDF outline extraction logic
- `Collection/pdf/` — Place your input PDFs here
- `Collection/Outputgoof2/` — Intermediate outputs and logs
- `requirements.txt` — Python dependencies
- `Dockerfile` — For containerized execution

## Customization
- Edit the `PERSONA` and `JOB_TO_BE_DONE` variables in `1bsol.py` to change the analysis focus.

## Troubleshooting
- Ensure your PDFs are not encrypted or password-protected.
- For GPU acceleration, ensure CUDA is available and supported by your hardware.

## License
MIT License

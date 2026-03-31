# HR Portal Backend & Frontend

## Overview
This project is a full-stack HR portal for resume management, candidate shortlisting, and interview question generation. It uses a Python FastAPI backend and a React frontend. The backend leverages AI (LLM) for job description and question generation, and uses vector search (FAISS) for semantic resume matching.

---

## Features
- Upload and parse PDF resumes
- Extract contact details and text from resumes
- Generate embeddings for semantic search
- Store and search resumes using FAISS vector database
- Shortlist candidates based on job description (JD) using ATS scoring
- Generate AI-powered summaries and interview questions
- Store and manage data in MongoDB
- React frontend for user interaction

---

## Backend Logic (Python/FastAPI)

### 1. Resume Parsing & Embedding
- **PDF resumes** are placed in the `backend/resumes/` folder.
- Each PDF is parsed for text using `extract_text_from_pdf` (PDFPlumber, OCR, etc.).
- Contact details are extracted using `extract_contact_details` (regex/NLP).
- Text is embedded into a vector using `create_embedding` (sentence-transformers).

### 2. Vector Database (FAISS)
- Embeddings are stored in a FAISS index for fast similarity search.
- The index is initialized at server startup (`initialize_vector_db`).
- A probe embedding determines the vector dimension.
- Resumes are indexed and mapped to file paths and text.

### 3. ATS Scoring
- When a JD is submitted, it is embedded and compared to all resume embeddings.
- Cosine similarity is calculated for each resume.
- ATS score = 80% cosine similarity + 20% keyword overlap (with a cap).
- Only candidates above a minimum score are shortlisted.

### 4. AI Summaries & Questions
- Summaries and interview questions are generated using an external LLM (OpenRouter API, Gemini model).
- The backend sends prompts and receives responses directly from the LLM.
- No internal endpoint is used for LLM calls; all are direct HTTP requests.

### 5. MongoDB Integration
- Resume data, candidate status, and generated content are stored in MongoDB.
- CRUD operations are handled asynchronously using Motor.

### 6. FastAPI Endpoints
- `/shortlist`: Shortlist candidates for a JD
- `/generate-questions`: Generate interview questions for a candidate
- `/generate-jd`: Generate a job description using LLM
- `/candidates`, `/jds`, `/resume/{resume_id}`: Manage and serve data

---

## Frontend Logic (React)
- Built with React and Vite.
- Components for uploading resumes, viewing candidates, generating JDs, and questions.
- Uses fetch API to interact with backend endpoints.
- Displays results, summaries, and allows status updates.

---

## Requirements

### Backend
- Python 3.10+
- FastAPI
- Uvicorn
- Motor (async MongoDB)
- numpy
- requests
- faiss-cpu
- sentence-transformers
- pdfplumber, easyocr, pypdfium2
- pydantic
- python-dotenv
- spaCy
- MongoDB (running instance)

Install dependencies:
```
pip install -r requirements.txt
```

### Frontend
- Node.js 16+
- npm or yarn
- React
- Vite
- lucide-react (icons)

Install dependencies:
```
cd frontend
npm install
```

---

## Running the Project

### 1. Start MongoDB
Make sure MongoDB is running locally or update the connection string in your backend config.

### 2. Start Backend
```
cd backend
uvicorn api_server:app --reload
```

### 3. Start Frontend
```
cd frontend
npm run dev
```

---

## Environment Variables
- Place your OpenRouter API key in a `.env` file in the backend root:
	```
	VITE_OPENROUTER_API_KEY=your_api_key_here
	```

---

## Code Structure
- `backend/api_server.py`: Main FastAPI app, endpoints, and logic
- `backend/parser/`: PDF and contact extraction
- `backend/embeddings/`: Embedding logic
- `backend/vector_db/`: FAISS index logic
- `backend/database/`: MongoDB logic
- `frontend/src/`: React components and features

---


---


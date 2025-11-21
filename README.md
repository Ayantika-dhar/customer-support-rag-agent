# SmartAssist: AI-Powered Customer Support Agent

## Overview
SmartAssist is an AI-powered customer support assistant built for the NeoStats AI Engineer Case Study. It answers product-related queries using a hybrid approach of Retrieval-Augmented Generation (RAG) and real-time web search. Internal documentation is used as the primary knowledge source, while web search is triggered for fresh or external queries. The system is deployed end-to-end on Streamlit Cloud.

## Live Application
Streamlit Deployment: https://customer-support-rag-agent-ly2bygn6ialwnqzidomkeq.streamlit.app/

## Repository Structure
.
├── app.py  
├── requirements.txt  
├── config/  
│   └── config.py  
├── models/  
│   ├── llm.py  
│   └── embeddings.py  
├── utils/  
│   ├── rag.py  
│   ├── search.py  
│   └── assistant.py  
└── data/  
    └── docs/  
        ├── faq.txt  
        ├── pricing.txt  
        └── integration.txt  

## Key Features

### 1. Retrieval-Augmented Generation (RAG)
- Loads internal FAQ, pricing, and integration documents.
- Chunks and embeds text using sentence-transformers (MiniLM-L6-v2).
- Performs cosine-similarity search to retrieve top-k relevant chunks.

### 2. Web Search Integration
- Uses Tavily API when internal docs do not sufficiently answer a query.
- Triggered by low similarity scores or keywords such as “outage”, “latest”, “status”.
- Returns structured results (title, snippet, URL) for LLM context.

### 3. LLM Reasoning with Groq
- Uses Groq’s Llama 3.1 8B Instant model.
- Combines internal document chunks and web results into a single context block.
- Produces grounded answers that remain faithful to retrieved sources.

### 4. Response Modes
- Concise Mode: Short, direct, skim-friendly responses.
- Detailed Mode: Step-by-step answers with elaboration and optional source mentions.

### 5. Source Attribution
- Expandable “Sources Used” component shows internal doc sources and web results.
- Improves transparency and reliability of answers.

### 6. Streamlit User Interface
- Chat interface with session history.
- Sidebar controls: response mode, build knowledge base, clear chat history.
- Error handling for missing KB or missing API keys.

## How It Works
1. User enters a query in the Streamlit chat UI.
2. System retrieves relevant document chunks via vector similarity search.
3. Routing logic decides whether to use web search.
4. Final context is built using internal docs and web results.
5. Groq LLM generates the final grounded response.
6. UI displays answer along with document and web sources.

## Local Setup Instructions

### 1. Clone the Repository
git clone <your-repository-url>  
cd <project-folder>

### 2. Create Virtual Environment
python -m venv .neoenv  
.\.neoenv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Set Environment Variables
set GROQ_API_KEY=your_groq_key  
set TAVILY_API_KEY=your_tavily_key  
set GROQ_MODEL_NAME=llama-3.1-8b-instant

### 5. Run the Application
streamlit run app.py

## Streamlit Cloud Deployment
1. Push project to GitHub.
2. Go to https://streamlit.io/cloud and create a new app.
3. Select your repo, branch, and main file (app.py).
4. Under App Settings → Secrets, add:
   GROQ_API_KEY="your_key"  
   GROQ_MODEL_NAME="llama-3.1-8b-instant"  
   TAVILY_API_KEY="your_key"
5. Deploy and test the application.

## Technologies Used
- Python  
- Streamlit  
- Groq LLM (llama-3.1-8b-instant)  
- Sentence Transformers (all-MiniLM-L6-v2)  
- Tavily Web Search API  
- LangChain Core Components

## Delpoyed app link : https://customer-support-rag-agent-ly2bygn6ialwnqzidomkeq.streamlit.app/

Author:
Ayantika Dhar
M.Tech CSE IIT Jodhpur

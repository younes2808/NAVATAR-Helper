# NAVATAR-Helper
## - health chatbot based on RAG framework

## Overview
The NAVATAR-Helper is a healthcare-focused chatbot developed using a Retrieval-Augmented Generation (RAG) framework. It answers NEET (Not in Education, Employment, or Training)-related questions in both Norwegian and English, ensuring responses are reliable and sourced.

## Key Features
- **Bilingual Support**: Responds in Norwegian and English.
- **NEET-Specific Domain**: Provides responses limited to NEET topics.
- **Minimized Hallucinations**: Reduces incorrect outputs using strict source-based data retrieval.

## Technical Specifications
- **Embedding Models**: `Alibaba-NLP/gte-multilingual-base`
- **Vector Databases**: `Milvus Lite`
- **LLM Used**: `NorMistral-7B-warm-instruct`.
- **Frameworks and Libraries**:
  - **LangChain** for chaining components.
  - **Streamlit** for the user interface.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/sirin-koca/NAVATAR-Helper.git
   cd NAVATAR-Helper

## Project Overview 
- Handle health-related queries focusing on the NEET domain.
- Provide accurate, verifiable answers based on the provided dataset (PDF files).
- Minimize hallucinations by limiting answers to only content found in the NEET dataset.
- Support both English and Norwegian language queries, responding in the language of the question.

This project uses **Streamlit** for the frontend, and **LangChain**, **Milvus**, and **Huggingface models** for the backend, including embedding models to vectorize the PDF content.

## Features

- Multilingual Support (English and Norwegian)
- Retrieval-Augmented Generation (RAG) for accurate query responses
- Simple and intuitive UI built with Streamlit
- Tested with both RAG and non-RAG approaches for performance comparison
- Low hallucination rate by limiting responses to NEET-related data

### Prerequisites
- Python 3.8 or higher
- PyCharm or any Python IDE
- A virtual environment for dependency management
- Git for version control

---

OsloMet H2024 | Group Project






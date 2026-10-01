# Notion AI Assistant

An AI-powered chatbot built with Python and Streamlit that answers questions about Notion's products, features, pricing, plans, use cases, and FAQs.

The application uses a verified knowledge base containing information sourced from official Notion pages and Google Gemini for generating natural-language responses.

## Live Demo

[Notion AI Assistant](https://notion-ai-assistant.streamlit.app/)

## Features

- AI-powered question answering
- Verified Notion knowledge base
- Retrieval-based context selection
- Follow-up question support
- Notion plan and pricing information
- Product and feature information
- FAQ support
- Source links for answers
- Greeting and conversational responses
- Off-topic question handling
- Graceful API error handling
- Fallback responses when Gemini is unavailable
- Session message limit
- Clean Streamlit chat interface

## Technology Stack

- **Python** — Core programming language
- **Streamlit** — Web interface
- **Google Gemini API** — Natural-language response generation
- **JSON** — Knowledge base storage
- **pytest** — Automated testing
- **python-dotenv** — Environment variable management

## How It Works

The application follows a retrieval-based architecture:

```text
User Question
      ↓
Streamlit UI
      ↓
Chatbot Orchestrator
      ↓
Question Retrieval
      ↓
Relevant Knowledge Base Records
      ↓
Context + User Question
      ↓
Google Gemini
      ↓
Generated Answer
      ↓
Source Links

When a user asks a question, the application first searches the local Notion knowledge base for relevant information.
The most relevant records are then provided as context to Google Gemini. Gemini generates a natural-language response using the retrieved information.
If no relevant information is found, the chatbot does not ask Gemini to guess an answer. Instead, it provides a controlled fallback response.
This approach helps keep responses grounded in the verified information available in the knowledge base.
Knowledge Base
The knowledge base contains verified information about Notion, including:
- Company overview
- Products
- Features
- Notion AI
- Collaboration
- Pricing
- Plan differences
- Use cases
- FAQs
- Limitations
The knowledge base is stored in:
data/notion_kb.json

Each knowledge-base record contains structured fields such as:
id
category
topic
question
answer
keywords
source_url

Pricing records also contain additional information such as:
plan
displayed_price
billing_unit
pricing_status

The knowledge base uses information from official Notion sources so that the chatbot can provide verified and traceable answers.
Project Structure
notion_chatbot/
│
├── app.py
│
├── chatbot/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── config.py
│   ├── knowledge_base.py
│   ├── llm.py
│   ├── prompts.py
│   └── retriever.py
│
├── data/
│   └── notion_kb.json
│
├── tests/
│   ├── test_chatbot.py
│   ├── test_llm.py
│   ├── test_prompts.py
│   └── test_retriever.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

Important Files
app.py
Handles the Streamlit user interface, chat history, suggested questions, source links, and session controls.
chatbot/chatbot.py
Acts as the main orchestration layer between the user interface, retriever, prompts, and Gemini.
chatbot/retriever.py
Searches and ranks relevant knowledge-base records for the user's question.
chatbot/knowledge_base.py
Loads and validates the Notion knowledge base.
chatbot/prompts.py
Contains the system instructions, response templates, fallback messages, and suggested questions.
chatbot/llm.py
Handles communication with the Google Gemini API, including error handling and retries.
chatbot/config.py
Contains application configuration such as model settings, retrieval settings, limits, and environment configuration.
data/notion_kb.json
Contains the verified Notion knowledge base.
tests/
Contains automated tests for the retriever, prompts, Gemini integration, and chatbot orchestration.
Local Setup
1. Clone the repository
git clone https://github.com/pratik826/notion-ai-assistant.git
cd notion-ai-assistant

2. Create a virtual environment
python -m venv venv

3. Activate the virtual environment
Windows PowerShell:
.\venv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

5. Configure the Gemini API key
Create a .env file in the project root:
GEMINI_API_KEY=your_gemini_api_key_here

The .env file should never be committed to GitHub.
6. Run the application
streamlit run app.py

The application will open in your browser.
Running Tests
Run the complete automated test suite using:
python -m pytest -v

The test suite covers:
- Knowledge retrieval
- Pricing and plan queries
- Follow-up questions
- Prompt generation
- Gemini API handling
- Error handling
- Chatbot orchestration
- Fallback behavior
Security
The Gemini API key is not hard-coded into the application source code.
For local development, the API key is loaded from the .env file.
For deployment, the API key is stored using Streamlit Secrets.
The .gitignore file prevents sensitive files such as:
.env
.streamlit/secrets.toml

from being committed to GitHub.
The application UI also does not directly access or display the Gemini API key.
Limitations
- The chatbot can only provide information available in its verified knowledge base.
- It does not perform unrestricted web searches.
- Pricing and product information may change over time and should be verified against current official Notion information when necessary.
- Gemini responses depend on the availability and usage limits of the configured API service.
- The knowledge base needs to be updated when important company information changes.
- Retrieval performance depends on the keywords and information available in the knowledge base.
Future Improvements
Possible future improvements include:
- Automatic knowledge-base updates from official sources
- More advanced semantic retrieval
- Improved synonym and intent handling
- Conversation analytics
- Admin interface for knowledge-base management
- Authentication and user management
- Improved monitoring and logging
- Expanded test coverage
- Better handling of complex multi-part questions
Author
Pratik Redekar
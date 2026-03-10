
# AI Mobile Troubleshooting Assistant
An LLM-powered chatbot designed to help users troubleshoot smartphone issues using Retrieval-Augmented Generation (RAG).

## Features
- RAG architecture for context-aware answers
- FAISS vector search for fast document retrieval
- Groq LLM integration for intelligent responses
- Flask-based web interface
- FAQ document search with contextual citations

## Tech Stack
Python  
Flask  
FAISS  
Sentence Transformers  
Groq LLM API  

## Project Structure
AI-Mobile-Troubleshooting-Assistant
│
├── app.py
├── Chat_Bot.py
├── faq.txt
├── templates/
│   └── index.html
└── requirements.txt

## How to Run

1. Install dependencies
pip install -r requirements.txt

2. Set API key
set GROQ_API_KEY=your_key

3. Run the application
python app.py

4. Open browser
http://127.0.0.1:5000

## Use Case
The assistant helps users troubleshoot mobile device issues by retrieving relevant documentation and generating accurate responses using LLM reasoning.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

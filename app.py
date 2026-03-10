from flask import Flask, request, jsonify, render_template
from Chat_Bot import RAGIndex, load_docs_from_file, answer_with_llm
import os
from groq import Groq


FAQ_FILE = "faq.txt"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


app = Flask(__name__)


print("🔧 Loading documents and building FAISS index...")
raw_docs = load_docs_from_file(FAQ_FILE)
rag = RAGIndex()
rag.build(client, raw_docs)
print("✅ Chatbot Ready!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("message", "")
    if query is None:
        query = ""
    query = query.strip().lower()


    greetings = {"", "say hello", "hi", "hello", "hey", "hola"}
    if query in greetings:
        return jsonify({
            "reply": "👋 Hi there! I'm here to help you with your smartphone queries. Ask me anything!"
        })


    hits = rag.search(client, query)


    try:
        answer = answer_with_llm(client, query, hits)
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"reply": "⚠️ Sorry, there was a problem generating a response."})

    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(debug=True)

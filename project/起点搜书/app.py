from flask import Flask, request, jsonify
import json
from query_processor import preprocess_query
from rag_retriever import retrieve_books

app = Flask(__name__)

with open("data/books.json", "r") as f:
    books = json.load(f)


@app.route("/recommend", methods=["POST"])
def recommend():
    query = request.json.get("query")
    processed_query = preprocess_query(query)
    if not processed_query:
        return jsonify([])
    results = retrieve_books(processed_query, books)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)

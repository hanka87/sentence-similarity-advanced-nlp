Description:
A FastAPI-based web application that calculates the similarity between two input sentences using sentence transformers.
Working:



The application takes two input sentences from the user.

It uses the sentence-transformers library to generate embeddings for each sentence.
The weighted similarity between the two embeddings is calculated.
The similarity score is returne
d to the user.
Technical Specifications:
Backend Framework: FastAPI
Language Model: Sentence Transformers (all-MiniLM-L6-v2)
Library: sentence-transformers, numpy, torch
Deployment: render
Ultimate Goal:
The ultimate goal of this project is to provide a simple and efficient way to calculate sentence similarity, which can be used in various natural language processing applications such as:
Text classification
Information retrieval
Question answering
Chatbots
Features:
Calculates cosine similarity between two input sentences
Uses state-of-the-art sentence transformer models
Fast and efficient API endpoint
Future Plans:
Integrate with other NLP models for more advanced features
Improve the user interface for better user experience
Expand the application to support multiple languages
Installation and Usage:
Clone the repository: git clone https://github.com/your-username/your-repo-name.git
Install the requirements: pip install -r requirements.txt
Run the application: uvicorn main:app --host 0.0.0.0 --port 8000
Access the application: http://localhost:8000
API Documentation:
The API documentation is available at http://localhost:8000/docs
Contributing:
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.
License:
This project is licensed under the MIT License. See LICENSE for details.
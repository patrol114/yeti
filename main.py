import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from flask import Flask, request, render_template
from flask_limiter import Limiter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from re import sub

# Set up Flask app and Limiter
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Set up models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
llama_tokenizer = AutoTokenizer.from_pretrained('jarradh/llama2_70b_chat_uncensored')

# Define a function to generate responses using both models
def generate_response(user_input):
    # Check if llama_tokenizer is not None
    if llama_tokenizer is not None:
        # Preprocess user input
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Dodaj konwersję do ciągu
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Convert tokens back to string
        user_input_str = " ".join(tokens)

        # Generate response using LLM
        llama_input = user_input_str
        if llama_tokenizer is not None:
            llama_output = llama_tokenizer(llama_input, return_tensors="pt", truncation=True, max_length=512)
            if llama_output is not None:
                llama_output = llama_output.tolist()

        # Generate response using GPT-2
        gpt2_input = user_input_str
        if gpt2_input is not None:
            gpt2_output = gpt2_tokenizer(gpt2_input, return_tensors="pt", truncation=True, max_length=512)
            if gpt2_output is not None:
                gpt2_output = gpt2_output.tolist()

        # Combine and return the final response
        if llama_output is not None and gpt2_output is not None:
            response = llama_output + " " + gpt2_output
        else:
            response = None
    else:
        response = None
    return response

# Set up routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Rate limiting
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('message')
        response = generate_response(user_input)
        return response
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

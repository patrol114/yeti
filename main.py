import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from re import sub
import json

# Set up Flask app and Limiter
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Set up models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
llama_tokenizer = AutoTokenizer.from_pretrained('jarradh/llama2_70b_chat_uncensored')

def generate_response(user_input):
    # Check if llama_tokenizer is not None
    if llama_tokenizer is not None:
        # Process user input
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Add conversion to string
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
            if llama_output is not None and 'input_ids' in llama_output:
                # Use 'input_ids' as input to GPT-2 model
                llama_input_ids = llama_output['input_ids']
                gpt2_output = gpt2_model.generate(llama_input_ids, max_length=512, num_return_sequences=1)
                response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)  # Decode the tensor to text
            else:
                response = None
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
        user_input = request.get_json().get('user_input', '')
        response = generate_response(user_input)
        return jsonify({'response': response})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

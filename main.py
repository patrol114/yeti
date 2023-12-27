import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
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
llama_model = AutoModel.from_pretrained('TheBloke/Llama-2-70B-GPTQ')
llama_tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-70B-GPTQ')
bert_model = AutoModel.from_pretrained('bert-base')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base')

def generate_response(user_input, decoding_strategy="greedy", output_length=512):
    try:
        # Process user input with BERT tokenizer
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Add conversion to string
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Convert tokens back to string
        user_input_str = " ".join(tokens)

        # Process input with BERT model to understand user input
        bert_input = bert_tokenizer(user_input_str, return_tensors="pt", truncation=True, max_length=512)
        bert_output = bert_model(**bert_input)

        # Generate response using Llama
        llama_output = llama_model(bert_output.last_hidden_state)
        llama_output_text = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)

        # Process Llama output with GPT-2
        gpt2_input = gpt2_tokenizer(llama_output_text, return_tensors="pt", truncation=True, max_length=512)

        # Apply decoding strategy
        if decoding_strategy == "greedy":
            gpt2_output = gpt2_model.generate(gpt2_input.input_ids, max_length=output_length, num_return_sequences=1)
        elif decoding_strategy == "beam":
            gpt2_output = gpt2_model.generate(gpt2_input.input_ids, max_length=output_length, num_return_sequences=1,
                                              num_beams=5)
        elif decoding_strategy == "top-k":
            gpt2_output = gpt2_model.generate(gpt2_input.input_ids, do_sample=True, max_length=output_length,
                                              top_k=50)
        elif decoding_strategy == "top-p":
            gpt2_output = gpt2_model.generate(gpt2_input.input_ids, do_sample=True, max_length=output_length,
                                              top_p=0.95)

        response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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

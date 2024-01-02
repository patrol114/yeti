import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, MarianMTModel, MarianTokenizer, AutoModel
from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from re import sub
import json
import os

# Setting environment variable
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Flask app and Limiter settings
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Setting GPU memory usage for PyTorch
torch.backends.cuda.reserved_megabytes = 512
torch.backends.cuda.max_split_size_mb = 512
token = "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"

# Initializing models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
llama_model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-13B-GPTQ', device_map="auto", trust_remote_code=False, revision="main")
llama_tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-GPTQ', use_fast=True)
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
translator_model_name = 'Helsinki-NLP/opus-mt-pl-en'  # Model for translation to Polish
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name, token=token)
translator_model = MarianMTModel.from_pretrained(translator_model_name, token=token)

# Function to generate response based on user input
def generate_response(user_input, decoding_strategy="greedy", output_length=512, translate_to_polish=False):
    try:
        # Processing user input with BERT tokenizer
        user_input = sub(r'[^\w\s]', '', str(user_input))
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Converting tokens back to a string
        user_input_str = " ".join(tokens)

        # Processing input with BERT model
        bert_input = bert_tokenizer(user_input_str, return_tensors="pt", truncation=True, max_length=512)
        bert_output = bert_model(**bert_input)

        # Generating response using Llama
        llama_input = llama_tokenizer.encode(user_input_str, return_tensors='pt')
        # Adding attention mask and pad_token_id
        llama_output = llama_model.generate(llama_input, max_length=200, pad_token_id=50256, attention_mask=llama_input)
        llama_output_decoded = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)

        # Processing Llama result with GPT-2 model
        gpt2_input = gpt2_tokenizer.encode(llama_output_decoded, return_tensors="pt")
        gpt2_input = gpt2_input.to(gpt2_model.device)

        # Applying decoding strategy
        if decoding_strategy == "greedy":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)
        elif decoding_strategy == "beam":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1, num_beams=5)
        elif decoding_strategy == "top-k":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_k=50)
        elif decoding_strategy == "top-p":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_p=0.95)
        else:  # default strategy
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)

        response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        # Translating the response to Polish if required
        if translate_to_polish:
            translation_tokens = translator_tokenizer.prepare_seq2seq_batch([response], return_tensors="pt")
            translated_output = translator_model.generate(**translation_tokens)
            response = translator_tokenizer.decode(translated_output[0], skip_special_tokens=True)

        return response

    except Exception as e:
        print(f"Error encountered: {e}")
        return None

# Route settings
@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Limiting the number of requests
def chatbot():
    if request.method == 'POST':
        # Extracting user input from JSON object
        user_input = request.get_json().get('user_input', '')
        if not isinstance(user_input, str):
            # Error handling, e.g., throwing an exception or returning an error message
            print("Error: User input is not text.")
            return jsonify({'response': 'Error: User input is not text.'})

        decoding_strategy = request.get_json().get('decoding_strategy', '')
        translate_to_polish = request.get_json().get('translate_to_polish', False)

        # Generating response using the updated generate_response function
        response = generate_response(user_input, decoding_strategy, translate_to_polish)

        # Return response as JSON
        return jsonify({'response': response})
    else:
        return render_template('index1.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

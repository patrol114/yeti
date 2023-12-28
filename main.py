import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
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
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
llama_model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-70B-GPTQ',
                                                   device_map="auto",
                                                   trust_remote_code=False,
                                                   revision="main")
llama_tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-70B-GPTQ', use_fast=True)
bert_model = AutoModel.from_pretrained('bert-base')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base')

def generate_response(user_input, decoding_strategy="greedy", output_length=512, translate_to_polish=False):
    try:
        # Process user input with BERT tokenizer
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Adding conversion to string
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Converting tokens back to a string
        user_input_str = " ".join(tokens)

        # Process input with BERT model
        bert_input = bert_tokenizer(user_input_str, return_tensors="pt", truncation=True, max_length=512)
        bert_output = bert_model(**bert_input)

        # Generate responses using Llama and GPT-2
        llama_input = llama_tokenizer.encode(user_input_str, return_tensors='pt')
        llama_output = llama_model.generate(llama_input, max_length=200, pad_token_id=50256, attention_mask=llama_input)
        llama_output_decoded = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)

        gpt2_input = gpt2_tokenizer.encode(llama_output_decoded, return_tensors="pt")
        gpt2_input = gpt2_input.to(gpt2_model.device)
        gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)
        gpt2_output_decoded = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        # Collect predictions from all three models
        predictions = [bert_output.argmax().item(), llama_output[0].argmax().item(), gpt2_output[0].argmax().item()]

        # Determine the final response using majority voting
        final_prediction = max(set(predictions), key=predictions.count)

        # Generate the final response using the appropriate model
        if final_prediction == 0:
            final_response = bert_tokenizer.decode(bert_output[0], skip_special_tokens=True)
        elif final_prediction == 1:
            final_response = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)
        else:
            final_response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        # Translating the response to Polish if required
        if translate_to_polish:
            translation_tokens = translator_tokenizer.prepare_seq2seq_batch([final_response], return_tensors="pt")
            translated_output = translator_model.generate(**translation_tokens)
            final_response = translator_tokenizer.decode(translated_output[0], skip_special_tokens=True)

        return final_response

    except Exception as e:
        print(f"Error encountered: {e}")
        return None

# Set up routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Rate limiting
def chatbot():
    if request.method == 'POST':
        # Extract user input from JSON object
        user_input = request.get_json().get('user_input', '')

        # Generate response using the updated generate_response function
        response = generate_response(user_input)

        # Return the response in JSON format
        return jsonify({'response': response})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

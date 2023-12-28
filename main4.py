import os
import torch
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from re import sub

# Flask app and Limiter settings
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Set environment variables
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Initialize models and tokenizers
model_names = {
    'gpt2': 'gpt2-xl',
    'llama': 'TheBloke/Llama-2-13B-GPTQ',
    'bert': 'bert-base-uncased',
    'translator': 'Helsinki-NLP/opus-mt-pl-en'
}

models = {name: {'model': None, 'tokenizer': None} for name in model_names.keys()}

# Set up logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# Function to load models and tokenizers
def load_model_and_tokenizer(name):
    model_cls, tokenizer_cls = (AutoModelForCausalLM, AutoTokenizer) if name != 'bert' else (AutoModel, AutoTokenizer)
    models[name]['model'] = model_cls.from_pretrained(model_names[name])
    models[name]['tokenizer'] = tokenizer_cls.from_pretrained(model_names[name])

# Load models on demand
for name in model_names.keys():
    load_model_and_tokenizer(name)

def ensemble_predictions(user_input):
    # Preprocess input
    user_input = sub(r'[^\w\s]', '', user_input).lower()
    tokens = word_tokenize(user_input)
    stop_words = set(stopwords.words('english'))
    filtered_input = " ".join([token for token in tokens if token not in stop_words])

    # Get predictions from each model
    predictions = []
    for name, components in models.items():
        if name == 'translator':  # Skip translator for ensemble
            continue
        input_ids = components['tokenizer'].encode(filtered_input, return_tensors='pt')
        output = components['model'].generate(input_ids, max_length=200)
        decoded_output = components['tokenizer'].decode(output[0], skip_special_tokens=True)
        predictions.append(decoded_output)

    # Combine predictions (simple averaging ensemble for demonstration)
    combined_prediction = " ".join(predictions)
    return combined_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def chatbot():
    if request.method == 'POST':
        user_input = request.get_json().get('user_input', '')
        if not isinstance(user_input, str):
            logging.error("Error: User input is not text.")
            return jsonify({'response': 'Error: User input is not text.'})

        # Generate response using ensemble
        response = ensemble_predictions(user_input)

        # Translate the response to Polish if required
        translate_to_polish = request.get_json().get('translate_to_polish', False)
        if translate_to_polish:
            translation_tokens = models['translator']['tokenizer'].prepare_seq2seq_batch([response], return_tensors="pt")
            translated_output = models['translator']['model'].generate(**translation_tokens)
            response = models['translator']['tokenizer'].decode(translated_output[0], skip_special_tokens=True)

        return jsonify({'response': response})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)
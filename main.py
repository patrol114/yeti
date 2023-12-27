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

def generate_response(user_input):
    # Sprawdź, czy llama_tokenizer nie jest None
    if llama_tokenizer is not None:
        # Przetwórz wejście od użytkownika
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Dodaj konwersję do ciągu
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Konwertuj tokeny z powrotem do ciągu
        user_input_str = " ".join(tokens)

        # Generuj odpowiedź przy użyciu LLM
        llama_input = user_input_str
        if llama_tokenizer is not None:
            llama_output = llama_tokenizer(llama_input, return_tensors="pt", truncation=True, max_length=512)
            if llama_output is not None and 'input_ids' in llama_output:
                # Użyj 'input_ids' jako wejścia do modelu GPT-2
                llama_input_ids = llama_output['input_ids']
                gpt2_output = gpt2_model.generate(llama_input_ids, max_length=512, num_return_sequences=1)
                response = str(gpt2_output[0].item())  # Pobierz wartość tensora jako zwykłą liczbę Pythona i konwertuj na ciąg znaków
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
        user_input = request.form.get('message')
        response = generate_response(user_input)
        return response
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

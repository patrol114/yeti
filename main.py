import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from flask import Flask, request, render_template
from flask_limiter import Limiter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from re import sub
import ngrok

# Set up Flask app and Limiter
app = Flask(__name__)
limiter = Limiter(app)

# Konfiguracja ngrok
ngrok.set_auth_token('2ZVsqXN2HRckjOt9KsJOtP2ssMl_49B9spuCEtipJDUBXNTLo')
# Numer portu, na którym będzie działać aplikacja
port = 9875

# Utworzenie tunelu ngrok i pobranie publicznego URL
http_tunnel = ngrok.connect(port)
print("Publiczny URL:", http_tunnel.public_url)

# Set up models and tokenizers
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
llama_tokenizer = AutoTokenizer.from_pretrained('jarradh/llama2_70b_chat_uncensored')

# Define a function to generate responses using both models
def generate_response(user_input):
    # Preprocess user input
    user_input = sub(r'[^\w\s]', '', user_input)
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Generate response using LLM
    llama_input = " ".join(tokens)
    llama_output = llama_tokenizer(llama_input, return_tensors="pt", truncation=True, max_length=512)
    llama_output = torch.tensor(llama_output)
    llama_preds = gpt2_model.generate(llama_output, max_length=512, num_return_sequences=1)
    llama_output = llama_preds[0][0].tolist()

    # Generate response using GPT-2
    gpt2_input = " ".join(tokens)
    gpt2_output = gpt2_tokenizer(gpt2_input, return_tensors="pt", truncation=True, max_length=512)
    gpt2_output = torch.tensor(gpt2_output)
    gpt2_preds = gpt2_model.generate(gpt2_output, max_length=512, num_return_sequences=1)
    gpt2_output = gpt2_preds[0][0].tolist()

    # Combine and return the final response
    response = llama_output + " " + gpt2_output
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
        return render_template('chatbot.html')

if __name__ == "__main__":
    app.run(port=port, debug=True)


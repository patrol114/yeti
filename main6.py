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

# Ustawienie zmiennej środowiskowej
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Konfiguracja aplikacji Flask i Limiter
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Ustawienia pamięci GPU dla PyTorch
torch.backends.cuda.reserved_megabytes = 512
torch.backends.cuda.max_split_size_mb = 512

# Inicjalizacja tokenizatorów
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
llama_tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-GPTQ', use_fast=True)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
translator_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-pl-en')

# Funkcja do generowania odpowiedzi na podstawie wejścia użytkownika
def generate_response(user_input, decoding_strategy="greedy", output_length=512, translate_to_polish=False):
    try:
        print(f"Input od użytkownika: {user_input}")
        # Przetwarzanie wejścia użytkownika za pomocą tokenizatora BERT
        user_input = sub(r'[^\w\s]', '', str(user_input))
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Konwersja tokenów z powrotem do stringa
        user_input_str = " ".join(tokens)
        input_ids = torch.tensor([1], device='cuda')  # DODANO DEKLARACJĘ input_ids

        # Przetwarzanie wejścia za pomocą modelu BERT
        bert_model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
        bert_input = bert_tokenizer(user_input_str, return_tensors="pt", truncation=True, max_length=512).to('cuda')
        bert_output = bert_model(**bert_input)
        print(f"Output od BERT: {bert_output}")

        # Generowanie odpowiedzi za pomocą Llama
        llama_model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-13B-GPTQ', device_map="auto", trust_remote_code=False, revision="main").to('cuda')
        llama_input = llama_tokenizer.encode(user_input_str, return_tensors='pt').to('cuda')
        # Dodawanie maski uwagi i pad_token_id
        llama_output = llama_model.generate(llama_input, max_length=100, pad_token_id=50256, attention_mask=llama_input)
        llama_output = llama_output_decoded = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)

        # Przetwarzanie wyniku z Llama za pomocą modelu GPT-2
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to('cuda')
        gpt2_input = gpt2_tokenizer.encode(llama_output_decoded, return_tensors="pt").to('cuda')

        # Zastosowanie strategii dekodowania
        if decoding_strategy == "greedy":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)
        elif decoding_strategy == "beam":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1, num_beams=5)
        elif decoding_strategy == "top-k":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_k=50)
        elif decoding_strategy == "top-p":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_p=0.95)
        else:  # strategia domyślna
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)

        response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        # Tłumaczenie odpowiedzi na polski, jeśli wymagane
        if translate_to_polish:
            translator_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-pl-en').to('cuda')
            translation_tokens = translator_tokenizer.prepare_seq2seq_batch([response], return_tensors="pt").to('cuda')
            translated_output = translator_model.generate(**translation_tokens)
            response = translator_tokenizer.decode(translated_output[0], skip_special_tokens=True)

        print(f"Odpowiedź wygenerowana: {response}")

        return response

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return None

# Ustawienia tras
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")  # Ograniczenie liczby żądań
def chatbot():
    if request.method == 'POST':
        # Wyciągnięcie wejścia użytkownika z obiektu JSON
        user_input = request.get_json().get('user_input', '')
        if not isinstance(user_input, str):
            # Obsługa błędu, np. rzucanie wyjątku lub zwracanie komunikatu o błędzie
            print("Błąd: Wejście użytkownika nie jest tekstem.")
            return jsonify({'response': 'Błąd: Wejście użytkownika nie jest tekstem.'})

        decoding_strategy = request.get_json().get('decoding_strategy', '')
        translate_to_polish = request.get_json().get('translate_to_polish', False)

        # Generowanie odpowiedzi za pomocą zaktualizowanej funkcji generate_response
        response = generate_response(user_input, decoding_strategy, translate_to_polish)

        # Zwrot odpowiedzi jako JSON
        return jsonify({'response': response})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

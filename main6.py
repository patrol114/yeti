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
import re


# Ustawienie zmiennej środowiskowej
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Konfiguracja aplikacji Flask i Limiter
app = Flask(__name__)
limiter = Limiter(app)
port = 9875

# Ustawienia pamięci GPU dla PyTorch
torch.backends.cuda.reserved_megabytes = 512
torch.backends.cuda.max_split_size_mb = 512

device_map = {0: [0,1,2,3,4,5,6,7,8,9,10,11,12], 1: [0,1,2,3,4,5,6,7,8,9,10,11,12]}

bert_model = AutoModel.from_pretrained('bert-base-uncased').to('cuda')
llama_model = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-13B-GPTQ', trust_remote_code=False, revision="main").to('cpu')

gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to('cuda')

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
llama_tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-13B-GPTQ', use_fast=True)
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
translator_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-pl-en')

# Funkcja do generowania odpowiedzi na podstawie wejścia użytkownika
def generate_response(user_input, decoding_strategy="greedy", translate_to_pl=False):
    try:
        # Inicjalizacja biblioteki NLTK
        nltk.download('punkt')
        nltk.download('stopwords')

        # Przetworzenie wejścia użytkownika za pomocą tokenizatora BERT
        cleaned_user_input = re.sub(r"[^\w\s]", "", str(user_input))
        lowered_user_input = cleaned_user_input.lower()
        tokens = word_tokenize(lowered_user_input)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]

        # Konwersja tokenów z powrotem na ciąg znaków
        context_tokens = " ".join(filtered_tokens)

        # Przetworzenie wejścia BERT
        bert_input = bert_tokenizer(context_tokens, return_tensors="pt", truncation=True, max_length=512).to('cuda')
        bert_output = bert_model(**bert_input)
        bert_output_decoded = bert_tokenizer.decode(bert_input["input_ids"][0], skip_special_tokens=True)
        print(f"Wyjście BERT: {bert_output_decoded}")

        # Generowanie odpowiedzi za pomocą modelu Llama
        llama_input = llama_tokenizer(bert_output_decoded, return_tensors="pt").to('cuda')

        # Dodanie maski uwagi i identyfikatora tokena pad
        attention_mask_llama = torch.ones_like(llama_input["input_ids"], device='cuda')
        llama_input["attention_mask"] = attention_mask_llama
        llama_input["decoder_input_ids"] = torch.ones_like(llama_input["input_ids"], device='cuda')
        llama_output = llama_model.generate(llama_input["input_ids"], max_length=512, pad_token_id=50256, attention_mask=attention_mask_llama)
        llama_output_decoded = llama_tokenizer.decode(llama_output[0], skip_special_tokens=True)

        print(f"Wyjście Generative LLAMA: {llama_output_decoded}")

        # Przetworzenie wyniku z modelu Llama za pomocą GPT-2
        gpt2_input = gpt2_tokenizer.encode(llama_output_decoded, return_tensors="pt").to('cuda')

        # Zastosowanie różnych strategii dekodowania w zależności od parametru
        output_length = 512  # Długość maksymalna odpowiedzi
        if decoding_strategy == "greedy":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)
        elif decoding_strategy == "beam":
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1, num_beams=5)
        elif decoding_strategy == "top-k":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_k=50)
        elif decoding_strategy == "top-p":
            gpt2_output = gpt2_model.generate(gpt2_input, do_sample=True, max_length=output_length, top_p=0.95)
        else:  # Strategia domyślna
            gpt2_output = gpt2_model.generate(gpt2_input, max_length=output_length, num_return_sequences=1)

        # Przetłumacz odpowiedź, jeśli wymagane
        if translate_to_pl:
            translator_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-pl-en').to('cuda')
            translation_tokens = translator_tokenizer.prepare_seq2seq_batch([gpt2_output[0]], return_tensors="pt").to('cuda')
            translated_output = translator_model.generate(**translation_tokens)
            response = translator_tokenizer.decode(translated_output[0], skip_special_tokens=True)
        else:
            response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)

        print(f"Generated response: {response}")
        return response

    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        return None


# Ustawienia tras
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def chatbot():
    if request.method == 'POST':
        # Wyciągnięcie wejścia użytkownika z obiektu JSON
        user_input = request.get_json().get('user_input', '')
        if not isinstance(user_input, str):
            # Obsługa błędu, np. rzucanie wyjątku lub zwracanie komunikatu o błędzie
            print("Błąd: Wejście użytkownika nie jest tekstem.")
            return jsonify({'response': 'Błąd: Wejście użytkownika nie jest tekstem.'})

        decoding_strategy = request.get_json().get('decoding_strategy', '')
        translate_to_pl = request.get_json().get('translate_to_pl', False)

        # Generowanie odpowiedzi za pomocą zaktualizowanej funkcji generate_response
        response = generate_response(user_input, decoding_strategy, translate_to_pl)

        # Zwrot odpowiedzi jako JSON
        return jsonify({'response': response})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=port, debug=True)

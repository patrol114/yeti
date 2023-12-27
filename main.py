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

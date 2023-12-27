def generate_response(user_input):
    # Check if llama_tokenizer is not None
    if llama_tokenizer is not None:
        # Preprocess user input
        user_input = sub(r'[^\w\s]', '', str(user_input))  # Dodaj konwersję do ciągu
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
                response = gpt2_output[0]
            else:
                response = None
        else:
            response = None
    else:
        response = None
    return response

# Imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama import Llama
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

#import ngrok
#from flask_ngrok import run_with_ngrok


# Load pre-trained GPT-2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
llama = Llama("lstm-large", tokenizer=gpt2_tokenizer)


# Flask App Initialization
app = Flask(__name__)
limiter = Limiter(app=app, key_func=get_remote_address)
# Setup ngrok
#ngrok.set_auth_token('2ZVsqXN2HRckjOt9KsJOtP2ssMl_49B9spuCEtipJDUBXNTLo')
#ngrok_tunnel = ngrok.connect(5601)

# Pobierz publiczny adres URL z ngrok_tunnel
#public_url = ngrok_tunnel.public_url if hasattr(ngrok_tunnel, 'public_url') else "N/A"

#print(public_url)
#run_with_ngrok(app)

# Routing for index.html
@app.route("/chatbot", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Jeśli to żądanie GET, zwróć stronę HTML
        return render_template("index.html")
    elif request.method == "POST":
        # Jeśli to żądanie POST, przetwórz dane i zwróć odpowiedź
        try:
            data = request.get_json()
            user_input = data.get("user_input")
            if user_input is None:
                return jsonify({"error": "User input is missing."}), 400

            bot_response = get_response(user_input)
            return jsonify({"response": bot_response})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Function to generate bot response
def get_response(user_input):
    # Get LLama response with max_gen_len set to 50 (możesz dostosować wartość według potrzeb)
    llama_resp = llama.generate(user_input, max_gen_len=50)

    # Get LLama response
    llama_resp = llama.generate(user_input)

    # Ensure that llama_resp is a string
    if isinstance(llama_resp, str):
        # Handle the case when llama_resp is a string
        # (you may need to adjust this based on llama's behavior)
        llama_params = llama_resp.split()
    else:
        # Handle the case when llama_resp is not a string or doesn't contain params
        return jsonify({"error": "Invalid response from LLama"}), 500

    # Generate GPT-2 response
    gpt2_input = gpt2_tokenizer.encode(llama_resp, return_tensors="pt")
    gpt2_output = gpt2_model.generate(gpt2_input, max_length=100)

    # Post-process GPT-2 response
    response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
    return response


# Execution
if __name__ == "__main__":
    app.run(debug=True, port=9875)

# Imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from llama import Llama
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import ngrok
from flask_ngrok import run_with_ngrok


# Load pre-trained GPT-2 model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
#llama_tokenizer = gpt2_tokenizer
llama = Llama("lstm-large")


# Flask App Initialization
app = Flask(__name__)
limiter = Limiter(app=app, key_func=get_remote_address)
# Setup ngrok
ngrok_tunnel = ngrok.connect(5000)
print('Public URL:', ngrok_tunnel.public_url)

run_with_ngrok(app)

# Chatbot Functionality
@app.route("/chatbot", methods=["POST"])
@limiter.limit("5 per minute")  # Rate limiting
def chat():
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
    # Get LLama response
    llama_resp = llama.predict(user_input)

    # Generate GPT-2 response
    gpt2_input = gpt2_tokenizer.encode(llama_resp, return_tensors="pt")
    gpt2_output = gpt2_model.generate(gpt2_input, max_length=100)

    # Post-process GPT-2 response
    response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
    return response

# Execution
if __name__ == "__main__":
    app.run(debug=True)

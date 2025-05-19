from flask import Flask, render_template, request, jsonify
import torch
from model.encoder import Encoder
from model.decoder import Decoder
from model.vocab import Vocab
from utils.preprocess import clean_text, create_mask

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab & models
vocab = Vocab()
vocab.load("model/vocab.pkl")  # Save/load your vocab as a pickle

encoder = Encoder(len(vocab.word2idx), 300, 512).to(device)
decoder = Decoder(len(vocab.word2idx), 300, 512).to(device)

checkpoint = torch.load("model/chatbot_model.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

def generate_response(text):
    cleaned = clean_text(text)
    input_seq = torch.tensor([vocab.encode(cleaned)], device=device)
    mask = create_mask(input_seq)

    with torch.no_grad():
        enc_outs, hidden = encoder(input_seq)
        input_token = torch.tensor([vocab.word2idx["<SOS>"]], device=device)
        decoded = []

        for _ in range(20):
            output, hidden, _ = decoder(input_token, hidden, enc_outs, mask)
            top1 = output.argmax(1).item()
            if top1 == vocab.word2idx["<EOS>"]:
                break
            decoded.append(top1)
            input_token = torch.tensor([top1], device=device)

        return vocab.decode(decoded)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["message"]
        response = generate_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    response = generate_response(data["message"])
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

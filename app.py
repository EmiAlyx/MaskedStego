from flask import Flask, render_template, request, jsonify
from masked_stego import MaskedStego, generate_text_gpt2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', "Once upon a time")
    max_length = int(data.get('max_length', 100))
    temperature = float(data.get('temperature', 0.7))
    top_p = float(data.get('top_p', 0.9))
    generated_text = generate_text_gpt2(prompt, max_length, temperature, top_p)
    return jsonify({'generated_text': generated_text})

@app.route('/encode', methods=['POST'])
def encode():
    data = request.get_json()
    cover_text = data.get('cover_text')
    message = data.get('message')
    mask_interval = int(data.get('mask_interval', 3))
    score_threshold = float(data.get('score_threshold', 0.01))

    masked_stego = MaskedStego()
    result = masked_stego(cover_text, message, mask_interval, score_threshold)
    return jsonify(result)

@app.route('/decode', methods=['POST'])
def decode():
    data = request.get_json()
    stego_text = data.get('stego_text')
    mask_interval = int(data.get('mask_interval', 3))
    score_threshold = float(data.get('score_threshold', 0.005))

    masked_stego = MaskedStego()
    result = masked_stego.decode(stego_text, mask_interval, score_threshold)
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)
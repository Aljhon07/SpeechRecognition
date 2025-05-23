from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow requests from your mobile app
import os
from datetime import datetime
from tools import audio
from inference.inference import inference
import config

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/transcribe', methods=['POST'])
def transcibe():
    
    print(f"Received request: {request.files}")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    print(f"Received audio file: {audio_file}")
    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'}), 400

    try:
        file_name = datetime.now().strftime('%I_%M_%S_%p')
        save_path = os.path.join(config.UPLOAD_DIR, file_name+'.m4a')
        audio_file.save(save_path)
        audio_file = audio.to_wav(save_path, save_path.replace('.m4a', '.wav'))        
        os.remove(save_path)

        transcript = inference(audio_file)
        print(f"Transcription result: {transcript}")

        
        return jsonify({'transcript': transcript}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
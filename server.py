from flask import Flask, request, jsonify
import subprocess
import threading
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

def run_script(script, input_text=None):
    if input_text:
        process = subprocess.Popen(['python', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.communicate(input_text.encode())
    else:
        subprocess.run(['python', script])

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    threading.Thread(target=run_script, args=("capture.py", name)).start()
    return jsonify({"message": "Image capture started."})

@app.route('/train', methods=['POST'])
def train():
    threading.Thread(target=run_script, args=("train.py",)).start()
    return jsonify({"message": "Training started."})

@app.route('/recognize', methods=['POST'])
def recognize():
    threading.Thread(target=run_script, args=("recognize.py",)).start()
    return jsonify({"message": "Recognition started."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

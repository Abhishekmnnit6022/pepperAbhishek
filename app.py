
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file.stream).convert("RGB")
        
        image_array = np.array(image).astype(np.float32)
        input_data = np.expand_dims(image_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_index = int(np.argmax(output_data))
        predicted_label = labels[predicted_index]
        return jsonify({'class': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

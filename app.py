from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# โหลดโมเดลเมื่อเริ่มต้นแอป
with open('trained_model.model', 'rb') as f:
    model_data = pickle.load(f)

# หน้าแรกแสดง UI
@app.route('/')
def home():
    return render_template('index.html')

# API สำหรับพยากรณ์
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ดึงข้อมูลจากคำขอ
        input_data = request.json.get('input', [])
        
        if not input_data:
            return jsonify({"error": "No input provided"}), 400

        # ดึง weights และ biases จากโมเดล
        w01 = model_data['w01']
        b01 = model_data['b01']
        
        # คำนวณพยากรณ์ (ตัวอย่างการใช้งาน)
        predictions = [x * w01 + b01 for x in input_data]  # linear prediction example
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

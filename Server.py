from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# โหลดโมเดล KNN ที่เทรนเสร็จแล้ว
model = joblib.load('knn_model.pkl')

# API endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูล JSON จาก request
        data = request.get_json()
        if not data:
            return jsonify({"status": "Failed", "message": "No input data provided"}), 400

        # แปลงข้อมูลเป็น DataFrame
        input_df = pd.DataFrame([data])

        # ทำ Prediction
        prediction = model.predict(input_df)

        # ส่งผลลัพธ์กลับเป็น JSON
        return jsonify({"status": "OK", "prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"status": "Failed", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

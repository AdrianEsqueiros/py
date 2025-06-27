from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Permitir CORS

# Cargar el modelo entrenado
modelo = joblib.load(os.path.join(os.path.dirname(__file__), "StackingAnemia.pkl"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    nuevo_paciente = pd.DataFrame([{
        "EdadMeses": data.get("EdadMeses"),
        "AlturaREN": data.get("AlturaREN"),
        "Sexo": data.get("Sexo"),
        "Suplementacion": data.get("Suplementacion"),
        "Cred": data.get("Cred"),
        "Tipo_EESS": data.get("Tipo_EESS"),
        "Red_simple": data.get("Red_simple"),
        "Grupo_Edad": data.get("Grupo_Edad"),
        "Suppl_x_EdadGrupo": data.get("Suppl_x_EdadGrupo"),
        "Sexo_x_Juntos": data.get("Sexo_x_Juntos"),
        "Indice_social": data.get("Indice_social")
    }])

    prediccion = modelo.predict(nuevo_paciente)[0]
    prob_anemia = modelo.predict_proba(nuevo_paciente)[0][1]

    return jsonify({
        "¿Tiene anemia?": "Sí" if prediccion == 1 else "No",
        "Probabilidad de anemia": round(prob_anemia, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

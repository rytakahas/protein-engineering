from flask import Flask, request, jsonify
import base64
import pandas as pd
import io
from mutation_predictor import predict_from_fasta_and_experiment

app = Flask(__name__)

@app.route('/mutation_prediction', methods=['POST'])
def mutation_prediction():
    data = request.get_json()

    wt_fasta = data.get("wild_type_fasta")
    experimental_csv_b64 = data.get("experimental_csv_base64")

    if not wt_fasta:
        return jsonify({"error": "Missing wild_type_fasta"}), 400

    df_experiment = None
    if experimental_csv_b64:
        try:
            decoded = base64.b64decode(experimental_csv_b64)
            df_experiment = pd.read_csv(io.BytesIO(decoded), sep=None, engine='python')  # auto-separator
        except Exception as e:
            return jsonify({"error": f"Error decoding experimental file: {str(e)}"}), 400

    try:
        predictions = predict_from_fasta_and_experiment(wt_fasta, df_experiment)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "wild_type_length": len(wt_fasta.strip()),
        "num_experimental_mutants": df_experiment.shape[0] if df_experiment is not None else 0,
        "predictions": predictions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5050)


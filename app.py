import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Load the trained model ─────────────────────────────────────────────────
# These .pkl files must sit in the same folder as this app.py in the repo.
MODEL_PATH    = 'candle_model.pkl'
FEATURES_PATH = 'candle_features.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)   # the 28 features the model was trained on

print(f'Model loaded. Expects {len(feature_names)} features:')
print(feature_names)

# ── Threshold (single source of truth — change here, not in the extension) ──
CONFIDENCE_THRESHOLD = 0.65

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        incoming_features = payload.get('features', {})

        if not incoming_features:
            return jsonify({'error': 'No features provided'}), 400

        # Only use features the model actually expects, in trained order.
        # Missing required feature -> error, not a silent zero-fill.
        missing = [f for f in feature_names if f not in incoming_features]
        if missing:
            return jsonify({
                'error': f'Missing required features: {missing}',
                'hint': 'Extension feature set does not match trained model.'
            }), 400

        extra = [k for k in incoming_features if k not in feature_names]
        if extra:
            print(f'[predict] Ignoring unexpected features not in model: {extra}')

        X_live = np.array([[incoming_features[f] for f in feature_names]])

        prob_down, prob_up = model.predict_proba(X_live)[0]
        prob_up   = float(prob_up)
        prob_down = float(prob_down)

        if prob_up >= CONFIDENCE_THRESHOLD:
            action     = 'BUY'
            confidence = prob_up
        elif prob_down >= CONFIDENCE_THRESHOLD:
            action     = 'SELL'
            confidence = prob_down
        else:
            action     = 'NO_TRADE'
            confidence = max(prob_up, prob_down)

        response = {
            'action':     action,
            'confidence': round(confidence, 4),
            'p_up':       round(prob_up, 4),
            'p_down':     round(prob_down, 4),
            'threshold':  CONFIDENCE_THRESHOLD,
            'reason':     None if action != 'NO_TRADE' else
                          f'Confidence {confidence*100:.1f}% below threshold {CONFIDENCE_THRESHOLD*100:.0f}%'
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_features_expected': len(feature_names),
        'threshold': CONFIDENCE_THRESHOLD
    })


# Render sets the PORT env var; gunicorn reads it via the start command,
# this block only matters if you ever run `python app.py` directly.
if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
          

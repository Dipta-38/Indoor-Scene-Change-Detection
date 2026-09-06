"""
Indoor Scene Change Detection -- Flask web interface.

Upload a BEFORE and AFTER photo of the same room; the app runs the full
camera-shift-aware detection + classification pipeline (identical logic to
the training notebook) and reports Add / Delete / Move with annotated images.

Setup:
    1. From the notebook, run the "Export for Flask Deployment" section
       (Section 11b) and download flask_model_bundle.zip.
    2. Unzip its contents into ./model/ so you have:
         model/best.pt
         model/rf_classifier.joblib
         model/config.json
    3. pip install -r requirements.txt
    4. python app.py
    5. Open http://localhost:5000
"""
import os
import uuid

import cv2
from flask import Flask, render_template, request, url_for, redirect, flash

from change_detection import ChangeDetector, draw_boxes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB per upload

detector = None  # lazy-loaded on first request so `flask run` fails fast with a clear
                  # error if model/ isn't populated, instead of failing silently at import time


def get_detector():
    global detector
    if detector is None:
        required = ['best.pt', 'config.json']
        missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {MODEL_DIR}. Run the notebook's 'Export for Flask "
                f"Deployment' section and copy the files here — see the docstring at the "
                f"top of app.py.")
        detector = ChangeDetector(MODEL_DIR)
    return detector


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    before_file = request.files.get('before_image')
    after_file = request.files.get('after_image')

    if not before_file or not after_file or before_file.filename == '' or after_file.filename == '':
        flash('Please choose both a BEFORE and an AFTER image.')
        return redirect(url_for('index'))
    if not (allowed_file(before_file.filename) and allowed_file(after_file.filename)):
        flash('Only .jpg, .jpeg, .png, .bmp files are supported.')
        return redirect(url_for('index'))

    session_id = uuid.uuid4().hex[:12]
    before_path = os.path.join(UPLOAD_DIR, f'{session_id}_before{os.path.splitext(before_file.filename)[1]}')
    after_path = os.path.join(UPLOAD_DIR, f'{session_id}_after{os.path.splitext(after_file.filename)[1]}')
    before_file.save(before_path)
    after_file.save(after_path)

    try:
        det = get_detector()
        result = det.analyze(before_path, after_path, tmp_dir=UPLOAD_DIR)
    except Exception as e:
        flash(f'Analysis failed: {e}')
        return redirect(url_for('index'))

    # Draw detections on the (already CLAHE-equalized) images actually seen by the model,
    # so what the user sees matches what drove the prediction.
    before_img = cv2.imread(result['before_eq_path'])
    after_img = cv2.imread(result['after_eq_path'])
    before_annot = draw_boxes(before_img, result['before_boxes'], color=(255, 140, 0))
    after_annot = draw_boxes(after_img, result['after_boxes'], color=(0, 200, 0))

    before_out = f'{session_id}_before_annot.jpg'
    after_out = f'{session_id}_after_annot.jpg'
    cv2.imwrite(os.path.join(UPLOAD_DIR, before_out), before_annot)
    cv2.imwrite(os.path.join(UPLOAD_DIR, after_out), after_annot)

    return render_template(
        'result.html',
        before_img=url_for('static', filename=f'uploads/{before_out}'),
        after_img=url_for('static', filename=f'uploads/{after_out}'),
        predicted_change=result['predicted_change'],
        predicted_object=result['predicted_object'],
        model_used=result['model_used'],
        camera_shift_residual_px=round(result['camera_shift_residual_px'], 2),
        move_thresh_used=round(result['move_thresh_used'], 2),
        stats=result['stats'],
        n_before=len(result['before_boxes']),
        n_after=len(result['after_boxes']),
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

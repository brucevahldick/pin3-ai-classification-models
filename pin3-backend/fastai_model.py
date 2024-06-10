# fastai_model.py
from fastai.vision.all import load_learner, PILImage
import io
import base64

# Load the FastAI model
learn = load_learner('fastai_model.pkl')


def get_fastai_prediction(image_bytes):
    img = PILImage.create(io.BytesIO(image_bytes))
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class

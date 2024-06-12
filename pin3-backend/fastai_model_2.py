from fastai.vision.all import *
from fastai.metrics import Precision, Recall
from pathlib import Path
import io

path = Path('../pin3-backend/resources/data')
model_path = '../pin3-backend/resources/data/models/model_fastai.pkl'


def get_fastai_data_loaders(batch_size=64):
    data_loaders = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224),
        bs=batch_size
    )
    return data_loaders


def train_and_save_model(epochs=3, lr=1e-3, batch_size=64):
    data_loaders = get_fastai_data_loaders(batch_size)
    learn = vision_learner(data_loaders, resnet34, metrics=[accuracy, Precision(), Recall()])

    # Fine-tune the model and display progress
    learn.fine_tune(epochs, base_lr=lr)

    learn.export(model_path)
    print(f'Model trained and saved to {model_path}')


def get_fastai_prediction(image_bytes):
    learn = load_learner(model_path)
    img = PILImage.create(io.BytesIO(image_bytes))
    pred_class, pred_idx, outputs = learn.predict(img)
    return pred_class


def evaluate_and_retrain_model(epochs=3, accuracy_threshold=0.9, batch_size=64):
    data_loaders = get_fastai_data_loaders(batch_size)
    accuracy = 0
    learn = vision_learner(data_loaders, resnet34, metrics=[accuracy, Precision(), Recall()])
    learn.load(model_path)

    # Evaluate the model
    accuracy = learn.validate()[1].item()

    # Retrain if necessary
    if accuracy < accuracy_threshold:
        learn.fine_tune(epochs)
        learn.export(model_path)

    precision = learn.validate()[2].item()
    recall = learn.validate()[3].item()
    best_params = learn.recorder.values[-1]

    return accuracy, precision, recall, best_params

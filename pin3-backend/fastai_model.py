import os
import torch
from fastai.vision.all import *
from PIL import Image
import io

path = Path('../pin3-backend/resources/data')
model_path = './resources/data/models/model_fastai.pth'


def train_and_save_model(epochs=6, lr=1e-2, batch_size=64, arch=resnet34):
    print("Loading data from dataset images: Start")
    dls = ImageDataLoaders.from_folder(
        path, train='train', valid='valid',
        item_tfms=Resize(128), bs=batch_size,
        batch_tfms=aug_transforms(do_flip=False)
    )
    print("Loading data from dataset images: Done")

    print("Starting training...")
    learn = vision_learner(dls, arch, metrics=[accuracy, Precision(average='macro'), Recall(average='macro')],
                           cbs=[MixedPrecision()])
    print("Done training...")

    if lr is None:
        lr_find_result = learn.lr_find()
        lr = lr_find_result.valley
        print(f"Using learning rate: {lr}")

    print("fit_one_cycle...")
    learn.fit_one_cycle(epochs, lr)

    print("unfreeze...")
    learn.unfreeze()
    print("fit_one_cycle...")
    learn.fit_one_cycle(epochs, lr)

    print("Create the directory if it does not exist...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Save the entire Learner object...")
    torch.save(learn, model_path)


def evaluate():
    print('Loading model...')
    learn = torch.load(model_path)
    print('model loaded.')

    val_loss, val_accuracy, val_precision, val_recall = learn.validate()
    print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, val_loss: {val_loss}")
    print(f"val_loss: {val_loss}")


def evaluate_and_retrain_model(max_iterations, target_accuracy, epochs, lr, batch_size):
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_params = {'epochs': epochs, 'lr': lr, 'batch_size': batch_size, 'arch': resnet34}

    if not os.path.exists(model_path):
        train_and_save_model()

    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations} with params: {best_params}")

        learn = torch.load(model_path)

        val_loss, val_metrics = learn.validate()
        val_accuracy = val_metrics[0]
        val_precision = val_metrics[1]
        val_recall = val_metrics[2]

        print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_precision = val_precision
            best_recall = val_recall
            best_params = {'epochs': best_params['epochs'] + 1,
                           'lr': best_params['lr'] * 0.9,
                           'batch_size': best_params['batch_size'],
                           'arch': best_params['arch']}
            torch.save(learn, model_path)
        else:
            best_params['lr'] *= 1.1

        if best_accuracy >= target_accuracy:
            break

    return best_accuracy, best_precision, best_recall, best_params


def get_fastai_prediction(image_bytes):
    learn = torch.load(model_path)

    try:
        img = PILImage.create(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    if img is None:
        raise ValueError("Loaded image is None")

    try:
        pred_class, pred_idx, outputs = learn.predict(img)
    except AttributeError as e:
        raise AttributeError(f"Error during prediction: {e}")

    return pred_class

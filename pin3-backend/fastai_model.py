import os
import torch
from fastai.vision.all import *
from PIL import Image
import io

# Definir o caminho para os dados e o nome do arquivo do modelo
path = Path('../pin3-backend/resources/data')
model_path = './resources/data/models/model_fastai.pth'


def train_and_save_model(epochs=6, lr=1e-2, batch_size=64, arch=resnet34):
    # Load data
    print("Loading data from dataset images: Start")
    dls = ImageDataLoaders.from_folder(
        path, train='train', valid='valid',
        item_tfms=Resize(128), bs=batch_size,
        batch_tfms=aug_transforms(do_flip=False)
    )
    print("Loading data from dataset images: Done")

    print("Starting training...")
    # Create the model using a pre-trained architecture
    learn = vision_learner(dls, arch, metrics=[accuracy, Precision(average='macro'), Recall(average='macro')], cbs=[MixedPrecision()])
    print("Done training...")

    # Find the best learning rate if not provided
    if lr is None:
        lr_find_result = learn.lr_find()
        lr = lr_find_result.valley  # Using the learning rate from the valley point
        print(f"Using learning rate: {lr}")

    print("fit_one_cycle...")
    # Train only the head for a few epochs, then unfreeze and continue training
    learn.fit_one_cycle(epochs, lr)

    print("unfreeze...")
    # Unfreeze the model and continue training for the specified number of epochs
    learn.unfreeze()
    print("fit_one_cycle...")
    learn.fit_one_cycle(epochs, lr)

    print("Create the directory if it does not exist...")
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Save the entire Learner object...")
    # Save the entire Learner object
    torch.save(learn, model_path)


def evaluate():
    # Load the entire Learner object
    print('Loading model...')
    learn = torch.load(model_path)
    print('model loaded.')

    # Avaliar o modelo atual
    val_loss, val_accuracy, val_precision, val_recall = learn.validate()
    # Print val_metrics for debugging
    # print(f"val_metrics: {val_metrics}")

    # val_accuracy = val_metrics[0] if len(val_metrics) > 0 else None
    # val_precision = val_metrics[1] if len(val_metrics) > 1 else None
    # val_recall = val_metrics[2] if len(val_metrics) > 2 else None
    print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, val_loss: {val_loss}")
    print(f"val_loss: {val_loss}")



def evaluate_and_retrain_model(max_iterations, target_accuracy):
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_params = {'epochs': 3, 'lr': 1e-1, 'batch_size': 64, 'arch': resnet34}

    if not os.path.exists(model_path):
        train_and_save_model()

    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations} with params: {best_params}")

        # Load the entire Learner object
        learn = torch.load(model_path)

        # Avaliar o modelo atual
        val_loss, val_metrics = learn.validate()
        val_accuracy = val_metrics[0]
        val_precision = val_metrics[1]
        val_recall = val_metrics[2]

        print(f"Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}")

        # Verificar se o modelo melhorou
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_precision = val_precision
            best_recall = val_recall
            best_params = {'epochs': best_params['epochs'] + 1,
                           'lr': best_params['lr'] * 0.9,
                           'batch_size': best_params['batch_size'],
                           'arch': best_params['arch']}
            # Save the improved Learner object
            torch.save(learn, model_path)
        else:
            # Tentar diferentes parâmetros se a acurácia não melhorar
            best_params['lr'] *= 1.1  # Aumentar a taxa de aprendizado

        # Parar se atingir a acurácia alvo
        if best_accuracy >= target_accuracy:
            break

    return best_accuracy, best_precision, best_recall, best_params


def get_fastai_prediction(image_bytes):
    # Load the entire Learner object
    learn = torch.load(model_path)


    # Try to open the image
    try:
        img = PILImage.create(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Ensure the image is not None
    if img is None:
        raise ValueError("Loaded image is None")

    # Perform prediction
    try:
        pred_class, pred_idx, outputs = learn.predict(img)
    except AttributeError as e:
        raise AttributeError(f"Error during prediction: {e}")

    # Retornar a classe prevista
    return pred_class

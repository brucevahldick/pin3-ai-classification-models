import os
from fastai.vision.all import *
from PIL import Image
import io

# Definir o caminho para os dados e o nome do arquivo do modelo
path = Path('../pin3-backend/resources/data')
model_path = 'model_fastai.pkl'

def train_and_save_model(epochs=3, lr=1e-1, batch_size=64, arch=resnet34):
    # Carregar os dados
    dls = ImageDataLoaders.from_folder(path, train='train', valid='valid', item_tfms=Resize(224), bs=batch_size, batch_tfms=aug_transforms())

    # Criar o modelo usando uma arquitetura pré-treinada
    learn = vision_learner(dls, arch, metrics=[accuracy, Precision(), Recall()])

    # Encontrar a melhor taxa de aprendizado
    learn.lr_find()
    
    # Treinar o modelo
    learn.fine_tune(epochs, lr)

    # Salvar o modelo treinado
    learn.export(model_path)

def evaluate_and_retrain_model(max_iterations, target_accuracy):
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_params = {'epochs': 3, 'lr': 1e-1, 'batch_size': 64, 'arch': resnet34}

    if not os.path.exists(model_path):
        train_and_save_model()

    for i in range(max_iterations):
        print(f"Iteration {i+1}/{max_iterations} with params: {best_params}")
        
        # Carregar o modelo treinado
        learn = load_learner(model_path)
        
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
        else:
            # Tentar diferentes parâmetros se a acurácia não melhorar
            best_params['lr'] *= 1.1  # Aumentar a taxa de aprendizado

        # Parar se atingir a acurácia alvo
        if best_accuracy >= target_accuracy:
            break

    return best_accuracy, best_precision, best_recall, best_params

def get_fastai_prediction(image_bytes):
    # Verificar se o modelo existe, se não, treinar e salvar
    if not os.path.exists(model_path):
        train_and_save_model()
    
    # Carregar o modelo treinado
    learn = load_learner(model_path)
    
    # Processar a imagem
    img = Image.open(io.BytesIO(image_bytes))
    
    # Realizar a previsão
    pred_class, pred_idx, outputs = learn.predict(img)
    
    # Retornar a classe prevista
    return pred_class
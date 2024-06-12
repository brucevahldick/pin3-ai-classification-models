import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fastai.vision.all import *
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Caminhos dos modelos e dados
data_path = Path('path_to_your_data')
model_path_fastai = 'model_fastai.pkl'
model_path_pytorch = 'model_pytorch.pth'

# Carregar os dados de teste
def load_test_data_pytorch(data_path, batch_size=64):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_path, 'test'), data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, test_dataset.classes

# Função para avaliar modelo FastAI
def evaluate_fastai_model(learn, test_loader):
    y_true = []
    y_pred = []
    for inputs, labels in test_loader:
        outputs = learn.predict(inputs)
        preds = torch.tensor([learn.dls.vocab[o[0]] for o in outputs])
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())
    return y_true, y_pred

# Função para avaliar modelo PyTorch
def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

# Função para calcular métricas
def calculate_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    return accuracy, precision, recall

# Função principal para carregar os modelos e avaliar
def main():
    # Carregar dados de teste
    test_loader, class_names = load_test_data_pytorch(data_path)
    
    # Avaliar modelo FastAI
    learn = load_learner(model_path_fastai)
    y_true_fastai, y_pred_fastai = evaluate_fastai_model(learn, test_loader)
    accuracy_fastai, precision_fastai, recall_fastai = calculate_metrics(y_true_fastai, y_pred_fastai, class_names)
    
    # Avaliar modelo PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model_pytorch(len(class_names))
    model.load_state_dict(torch.load(model_path_pytorch, map_location=device))
    model = model.to(device)
    y_true_pytorch, y_pred_pytorch = evaluate_pytorch_model(model, test_loader, device)
    accuracy_pytorch, precision_pytorch, recall_pytorch = calculate_metrics(y_true_pytorch, y_pred_pytorch, class_names)
    
    # Exibir resultados
    print("FastAI Model Results:")
    print(f"Accuracy: {accuracy_fastai:.4f}")
    print(f"Precision: {precision_fastai:.4f}")
    print(f"Recall: {recall_fastai:.4f}")
    
    print("\nPyTorch Model Results:")
    print(f"Accuracy: {accuracy_pytorch:.4f}")
    print(f"Precision: {precision_pytorch:.4f}")
    print(f"Recall: {recall_pytorch:.4f}")

# Função para inicializar modelo PyTorch (copiada do arquivo pytorch_model.py)
def initialize_model_pytorch(num_classes, feature_extract=True, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

if __name__ == '__main__':
    main()

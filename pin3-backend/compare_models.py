import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fastai.vision.all import *
from PIL import Image

path = Path('../pin3-backend/resources/data')
model_pytorch_path = '../pin3-backend/resources/data/models/model_pytorch.pth'
model_fastai_path = '../pin3-backend/resources/data/models/model_fastai.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_test_data(batch_size=64):
    test_dataset = datasets.ImageFolder(os.path.join(path, 'test'), data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    return test_loader, class_names


def initialize_model(num_classes):
    weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def evaluate_pytorch_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall


def evaluate_fastai_model(learn, test_loader):
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs = inputs.cpu().numpy()
        for i in range(inputs.shape[0]):
            img = Image.fromarray((inputs[i].transpose(1, 2, 0) * 255).astype('uint8'))
            pred_class, pred_idx, outputs = learn.predict(img)
            all_preds.append(pred_idx)
            all_labels.append(labels[i].item())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall


def main():
    test_loader, class_names = load_test_data()

    num_classes = len(class_names)
    model_pytorch = initialize_model(num_classes)
    model_pytorch.load_state_dict(torch.load(model_pytorch_path, map_location=device))
    model_pytorch = model_pytorch.to(device)
    pytorch_acc, pytorch_precision, pytorch_recall = evaluate_pytorch_model(model_pytorch, test_loader)
    print(
        f"PyTorch Model - Accuracy: {pytorch_acc:.4f}, Precision: {pytorch_precision:.4f}, Recall: {pytorch_recall:.4f}")

    learn = load_learner(model_fastai_path)
    fastai_acc, fastai_precision, fastai_recall = evaluate_fastai_model(learn, test_loader)
    print(f"FastAI Model - Accuracy: {fastai_acc:.4f}, Precision: {fastai_precision:.4f}, Recall: {fastai_recall:.4f}")


if __name__ == "__main__":
    main()

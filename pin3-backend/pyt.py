import os
import base64
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from flask import Flask, request, jsonify
from PIL import Image
import io
from pathlib import Path

# Definir o caminho para os dados e o nome do arquivo do modelo
path = Path('../pin3-backend/resources/data')
model_path = 'model_pytorch2.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(batch_size=64):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def initialize_model(num_classes, feature_extract=True, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloaders, dataset_sizes, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['valid']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    val_loss = running_loss / dataset_sizes['valid']
    val_acc = running_corrects.double() / dataset_sizes['valid']

    print(f'Valid Loss: {val_loss:.4f} Valid Acc: {val_acc:.4f}')

    return val_loss, val_acc


def retrain_model():
    dataloaders, dataset_sizes, class_names = load_data()
    num_classes = len(class_names)
    model = initialize_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_model = model

    best_loss, best_acc = evaluate_model(model, dataloaders, dataset_sizes, criterion)
    num_epochs = 25
    for epoch in range(num_epochs):
        print(f"Retraining with epoch {epoch + 1}/{num_epochs}")
        model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=1)
        val_loss, val_acc = evaluate_model(model, dataloaders, dataset_sizes, criterion)
        if val_acc > best_acc:
            best_model = model
            best_acc = val_acc
            best_loss = val_loss

    return best_model, best_acc, best_loss


def get_pytorch_prediction(image_bytes, class_names):
    # Carregar o modelo treinado
    model = initialize_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Processar a imagem
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]


def train_and_save_model(epochs=3, lr=1e-1, batch_size=64, arch=models.resnet34):
    # Load data
    dataloaders, dataset_sizes, class_names = load_data(batch_size=batch_size)

    # Initialize the model
    model = initialize_model(num_classes=len(class_names))
    model = model.to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=epochs)

    # Save the trained model
    torch.save(model.state_dict(), model_path)


def load_data(batch_size=64):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


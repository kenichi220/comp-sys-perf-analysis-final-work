# --- Bibliotecas para instalar no seu ambiente virtual (venv) ---
# pip install torch torchvision torchaudio

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# --- Etapa 1: Configurações e Seeds ---
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configura o dispositivo para GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}\n")


# --- Etapa 2: Carregamento e Pré-processamento dos Dados ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# --- Etapa 3: Construindo o Modelo ---
model = resnet50(weights=None) 

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10)
)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# ---Treinando o Modelo ---
print("Iniciando o Treinamento (com pesos aleatórios)...")
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Época [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


# --- Etapa 6: Avaliando o Desempenho ---
print("\nAvaliando o Modelo...")
model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_loss = test_loss / len(test_loader)
final_accuracy = 100 * correct / total

print(f"Loss (perda) no teste: {final_loss:.4f}")
print(f"Accuracy (acurácia) no teste: {final_accuracy:.2f}%")
print("\n--- Treinamento Concluído! ---")

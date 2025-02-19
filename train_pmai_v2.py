import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import glob
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # İlerleme çubuğu için
import json

import warnings
warnings.filterwarnings("ignore")

# CUDA ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Özel Dataset Sınıfı
class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for ext in ['jpg', 'png', 'jpeg']:
                    for img_path in glob.glob(f"{class_path}/*.{ext}"):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.to(device), torch.tensor(label, dtype=torch.long, device=device)

# Veri Yükleme ve Transform
root_dir = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\PlantMaster_Veriler\BIRLESTIRILMIS\VeriSetOrj224\veri_seti_yeni"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = PlantDiseaseDataset(root_dir, transform=transform)

# Train-Val Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader'lar
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # Batch boyutunu artırdık
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.resnet50(pretrained=True).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes)).to(device)

# Loss ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Eğitim Döngüsü
total_epochs = 1
best_val_loss = float('inf')
early_stopping_patience = 3
patience_counter = 0

if __name__ == '__main__':  # Ana program bloğu
    for epoch in range(total_epochs):
        # Eğitim
        model.train()
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        running_loss = 0.0

        for images, labels in train_progress:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_progress.set_postfix({"Loss": f"{loss.item():.4f}"})  # Anlık loss gösterimi

        # Validasyon
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
            for images, labels in val_progress:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_progress.set_postfix({"Val Loss": f"{loss.item():.4f}"})

        # Epoch Sonu Metrikler
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        # Konsola Detaylı Çıktı
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Accuracy:   {accuracy:.2f}%")

        # Early Stopping ve Model Kaydetme
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("  ↪ Yeni en iyi model kaydedildi!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nErken durdurma! Eğitim sonlandırıldı. En iyi Val Loss: {best_val_loss:.4f}")
                break

        scheduler.step()


    
    
    # Final Model Kaydetme
    torch.save(model.state_dict(), "final_model.pth")
    print("\nEğitim tamamlandı!")

    
    # Sınıf bilgilerini kaydet
    class_info = {
        "class_to_idx": dataset.class_to_idx,
        "classes": dataset.classes
    }
    with open("class_indices.json", "w") as f:
        json.dump(class_info, f)
    print("Sınıf bilgileri class_indices.json dosyasına kaydedildi.")
    
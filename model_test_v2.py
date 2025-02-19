import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import argparse

# Uyarıları kapat
import warnings
warnings.filterwarnings("ignore")

# Sabit değerler
MODEL_PATH = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\OrtakKodlar\final_model.pth"  # Eğitilmiş model dosyası
CLASS_INDICES_PATH = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\OrtakKodlar\class_indices.json"  # Sınıf isimlerinin kaydedildiği dosya

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıf bilgilerini yükle
def load_class_indices(class_indices_path):
    """Sınıf isimlerini ve indekslerini yükler."""
    with open(class_indices_path, "r") as f:
        class_info = json.load(f)
    return class_info["class_to_idx"], class_info["classes"]

# Modeli yükle
def load_model(model_path, num_classes, device):
    """Eğitilmiş modeli yükler."""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Resmi yükle ve tahmin yap
def predict(image_path, model, transform, classes, device):
    """Resmi yükler ve tahmin yapar."""
    # Resmi yükle ve transform uygula
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # En yüksek olasılıklı sınıfı bul
    confidence, pred_idx = torch.max(probabilities, 0)
    return classes[pred_idx.item()], confidence.item()

def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description='ResNet50 ile bitki hastalık tahmini')
    parser.add_argument('image_path', type=str, help='Tahmin edilecek resmin yolu')
    args = parser.parse_args()

    # Sınıf bilgilerini yükle
    class_to_idx, classes = load_class_indices(CLASS_INDICES_PATH)
    num_classes = len(classes)

    # Modeli yükle
    model = load_model(MODEL_PATH, num_classes, device)

    # Transform işlemleri
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Tahmin yap ve sonuçları göster
    class_name, confidence = predict(args.image_path, model, transform, classes, device)
    
    print("\n" + "="*50)
    print(f" Tahmin Sonuçları ".center(50, '='))
    print("="*50)
    print(f"Resim dosyası:     {args.image_path}")
    print(f"Tahmini sınıf:     {class_name}")
    print(f"Güvenilirlik:      {confidence*100:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
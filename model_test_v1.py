import argparse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Sabit değerler
MODEL_PATH = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\OrtakKodlar\best_model.pth"
DATASET_ROOT = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\PlantMaster_Veriler\BIRLESTIRILMIS\VeriSetOrj224\veri_seti_yeni"

# Tahmin için gerekli sınıf ve fonksiyonlar
class SimpleDataset:
    def __init__(self, root_dir):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

def load_model(model_path, num_classes, device):
    """Modeli yükleyip hazır hale getirir"""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(image_path, model, transform, classes, device):
    """Resmi yükleyip tahmin yapar"""
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

    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sınıf isimlerini yükle
    dataset = SimpleDataset(DATASET_ROOT)
    
    # Modeli yükle
    model = load_model(MODEL_PATH, len(dataset.classes), device)
    
    # Transform işlemleri
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Tahmin yap ve sonuçları göster
    class_name, confidence = predict(args.image_path, model, transform, dataset.classes, device)
    
    print("\n" + "="*50)
    print(f" Tahmin Sonuçları ".center(50, '='))
    print("="*50)
    print(f"Resim dosyası:     {args.image_path}")
    print(f"Tahmini sınıf:     {class_name}")
    print(f"Güvenilirlik:      {confidence*100:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
import tensorflow as tf
import numpy as np
import os
import cv2
import keyboard


model_path = "mobileNetPlantMaster.h5"
model = tf.keras.models.load_model(model_path)


dataset_path = r"C:\Users\Batuhan\Desktop\Resnet50modelegitim\veriler"
class_names = sorted(os.listdir(dataset_path))


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    class_name = class_names[predicted_class]
    return class_name, confidence


while True:
    image_path = input("Görüntü dosya yolunu girin (Çıkmak için 'esc' veya 'q' tuşuna basın): ")
    
    if image_path.lower() in ['esc', 'q']:
        print("Çıkış yapılıyor...")
        break
    
    if not os.path.exists(image_path):
        print("Hata: Dosya bulunamadı. Lütfen doğru yolu girin.")
        continue
    
    predicted_class, confidence = predict_image(image_path)
    print(f"Tahmin Edilen Sınıf: {predicted_class}")
    print(f"Güven Skoru: {confidence:.4f}\n")
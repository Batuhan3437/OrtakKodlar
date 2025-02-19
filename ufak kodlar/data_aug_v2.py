import os

from PIL import Image

import imgaug.augmenters as iaa

import numpy as np

import cv2
 
def get_image_count(directory):

    image_extensions = ('.png', '.jpg', '.jpeg')

    return sum(1 for file in os.listdir(directory) if file.lower().endswith(image_extensions))
 
def canny_augmentation(images, random_state, parents, hooks):

    return [cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 100, 200) for image in images]
 
def augment_images(directory, num_images_to_augment):

    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_files:

        print(f"{directory} dizininde resim bulunamadı.")

        return

    augmenters = [

        iaa.Fliplr(0.5), iaa.Affine(rotate=(-15, 15)), iaa.Multiply((0.8, 1.2)),

        iaa.GaussianBlur(sigma=(0, 1.0)), iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),

        iaa.LinearContrast((0.8, 1.2)), iaa.Affine(scale=(0.8, 1.2)), iaa.PerspectiveTransform(scale=(0.01, 0.1)),

        iaa.Sharpen(alpha=(0.0, 1.0)), iaa.Crop(percent=(0, 0.1)),

        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),

        iaa.LinearContrast((0.6, 1.4)), iaa.Resize({"height": (0.8, 1.2), "width": (0.8, 1.2)}),

        iaa.Rot90(1), iaa.Fliplr(0.7), iaa.Multiply((0.5, 1.5)), iaa.ElasticTransformation(alpha=(0.1, 0.3), sigma=0.25),

        iaa.PiecewiseAffine(scale=(0.01, 0.05)), iaa.Invert(0.1), iaa.Lambda(func_images=canny_augmentation),

        iaa.Dropout((0.01, 0.1)), iaa.Solarize(0.5), iaa.MotionBlur(k=5),

        iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(100, 1000)),

    ]

    for i, image_file in enumerate(image_files[:num_images_to_augment]):

        image_path = os.path.join(directory, image_file)

        with Image.open(image_path) as img:

            img_array = np.array(img.convert("RGB"))

            for j, augmenter in enumerate(augmenters):

                augmented_image = augmenter(image=img_array)

                aug_image = Image.fromarray(augmented_image)

                save_path = os.path.join(directory, f"{os.path.splitext(image_file)[0]}_aug_{j + 1}.jpg")

                aug_image.save(save_path)

                print(f"Kaydedildi: {save_path}")
 
def process_mosaic_virus(directory, desired_total=2500):

    if not os.path.exists(directory):

        print("Geçerli bir dizin giriniz.")

        return

    total_images = get_image_count(directory)

    print(f"{directory} dizininde toplam {total_images} resim bulundu.")

    if total_images >= desired_total:

        print(f"{directory} dizinindeki resim sayısı zaten yeterli.")

        return

    needed_images = desired_total - total_images

    augment_per_image = 25 

    images_to_augment = int(np.ceil(needed_images / augment_per_image))

    print(f"{directory} dizininde {images_to_augment} orijinal resim augmentasyon için seçilecek.")

    augment_images(directory, images_to_augment)
 
def main():

    while True:

        target_directory = input("Lütfen resimlerin bulunduğu klasör yolunu girin (Çıkmak için 'exit' yazın): ")

        if target_directory.lower() == 'exit':

            print("Çıkılıyor...")

            break

        process_mosaic_virus(target_directory)
 
if __name__ == "__main__":

    main()

 
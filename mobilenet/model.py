import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm  # İlerleme çubuğu için tqdm ekliyoruz

dataset_path = r"C:\Users\Batuhan\Desktop\Resnet50modelegitim\veriler"
train_path = "train"
test_path = "test"
val_path = "val"

classes = os.listdir(dataset_path)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    images = os.listdir(cls_path)
    

    train_imgs, test_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=test_ratio / (test_ratio + val_ratio), random_state=42)


    for img_set, set_name in tqdm([(train_imgs, train_path), (val_imgs, val_path), (test_imgs, test_path)], desc=f"Kopyalama: {cls}", total=3):
        set_class_path = os.path.join(set_name, cls)
        os.makedirs(set_class_path, exist_ok=True)  
        for img in img_set:
            src = os.path.join(cls_path, img)
            dest = os.path.join(set_class_path, img)
            
          
            if not os.path.exists(dest):
                shutil.copy(src, dest)


train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=16, class_mode='categorical')  # Batch size küçültüldü
val_generator = val_datagen.flow_from_directory(val_path, target_size=(224, 224), batch_size=16, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=16, class_mode='categorical')


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


for layer in base_model.layers[-30:]:
    layer.trainable = True  

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax') 
])


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


model_checkpoint = ModelCheckpoint('best_plant_disease_model_mobilenet.h5', monitor='val_loss', 
                                   save_best_only=True, mode='min', verbose=1)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Düşük learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint], 
    verbose=1  
)


test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

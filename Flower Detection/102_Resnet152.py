import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.applications import ResNet152
from tf_keras.applications.resnet import preprocess_input
from tf_keras import layers, models, optimizers
from tf_keras.callbacks import EarlyStopping

# 📁 Dosya yolları
IMG_PATH = 'data/flower102/jpg/'
LABELS_PATH = 'data/flower102/imagelabels.mat'
SETID_PATH = 'data/flower102/setid.mat'

# 🧪 Verileri yükle
labels = scipy.io.loadmat(LABELS_PATH)['labels'].flatten()
setid = scipy.io.loadmat(SETID_PATH)
val_ids = setid['trnid'].flatten()
test_ids = setid['valid'].flatten()
train_ids = setid['tstid'].flatten()

# 📷 Görsel yollarını hazırla
def get_image_paths_and_labels(image_ids, labels):
    image_paths = [os.path.join(IMG_PATH, f'image_{i:05d}.jpg') for i in image_ids]
    image_labels = [labels[i - 1] - 1 for i in image_ids]
    return image_paths, image_labels

train_paths, train_labels = get_image_paths_and_labels(train_ids, labels)
val_paths, val_labels = get_image_paths_and_labels(val_ids, labels)
test_paths, test_labels = get_image_paths_and_labels(test_ids, labels)

print("📊 Veri Seti Dağılımı:")
print(f"   🔹 Eğitim verisi:     {len(train_paths)} görsel")
print(f"   🔹 Doğrulama verisi:  {len(val_paths)} görsel")
print(f"   🔹 Test verisi:       {len(test_paths)} görsel")
print(f"   📁 Toplam veri:       {len(train_paths) + len(val_paths) + len(test_paths)} görsel")

# 🎨 Data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42
)
val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': val_paths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32,
    shuffle=False
)
test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_paths, 'class': test_labels}),
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

# 🧠 Model mimarisi
base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(102, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🔧 İlk eğitim (yalnızca dense katmanlar)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# 🔓 Fine-tuning (bazı ResNet katmanları da açılır)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stop]
)

# 🧮 Ortalama değer hesaplama
def print_avg_metrics(history_list, phase_name=""):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_list:
        acc.extend(h.history['accuracy'])
        val_acc.extend(h.history['val_accuracy'])
        loss.extend(h.history['loss'])
        val_loss.extend(h.history['val_loss'])

    print(f"\n📈 Ortalama Eğitim Sonuçları ({phase_name}):")
    print(f"   🔹 Eğitim Doğruluğu Ort:     {np.mean(acc):.4f}")
    print(f"   🔹 Eğitim Kaybı Ort:         {np.mean(loss):.4f}")
    print(f"   🔹 Doğrulama Doğruluğu Ort:  {np.mean(val_acc):.4f}")
    print(f"   🔹 Doğrulama Kaybı Ort:      {np.mean(val_loss):.4f}")

print_avg_metrics([history, fine_tune_history], "Tüm Eğitim")

# 📈 Eğitim grafikleri
def plot_training_history(histories):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc.extend(h.history['accuracy'])
        val_acc.extend(h.history['val_accuracy'])
        loss.extend(h.history['loss'])
        val_loss.extend(h.history['val_loss'])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Eğitim Doğruluğu')
    plt.plot(epochs_range, val_acc, label='Doğrulama Doğruluğu')
    plt.legend()
    plt.title('Doğruluk')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Eğitim Kaybı')
    plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı')
    plt.legend()
    plt.title('Kayıp')
    plt.show()

plot_training_history([history, fine_tune_history])

# 💾 Modeli kaydet
model.save("resnet152_flower_model.keras")
print("✅ Model 'resnet152_flower_model.keras' olarak kaydedildi.")

# 🧪 Test sonuçları
loss, acc = model.evaluate(test_generator)
print(f"\n🧪 Test Doğruluğu: {acc:.4f}")
print(f"📉 Test Kaybı: {loss:.4f}")

# 📊 Detaylı sınıf analizi
preds = model.predict(test_generator)
predicted_classes = np.argmax(preds, axis=1)
true_classes = np.array(test_generator.labels)

print("\n🧾 Sınıf Bazlı Performans Raporu:")
print(classification_report(true_classes, predicted_classes))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

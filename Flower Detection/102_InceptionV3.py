import numpy as np
import math
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tf_keras.applications import InceptionV3
from tf_keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Dosya Yolları
IMG_PATH = 'data/flower102/jpg/'
LABELS_PATH = 'data/flower102/imagelabels.mat'
SETID_PATH = 'data/flower102/setid.mat'


labels_data = scipy.io.loadmat(LABELS_PATH)
labels = labels_data['labels'].flatten()

setid_data = scipy.io.loadmat(SETID_PATH)
train_ids = setid_data['trnid'].flatten()
val_ids = setid_data['valid'].flatten()
test_ids = setid_data['tstid'].flatten()


def get_segmentation_paths(ids):
    paths = [os.path.join(IMG_PATH, f"image_{i:05d}.jpg") for i in ids]
    return [p for p in paths if os.path.exists(p)]

train_paths = get_segmentation_paths(train_ids)
val_paths = get_segmentation_paths(val_ids)
test_paths = get_segmentation_paths(test_ids)

# Burada dikkat: df_val ve df_test ile df_train isimleri yanlış atanmış gibi gözüküyor, 
# Eğitim için train_paths, val için val_paths, test için test_paths olmalı:
df_train = pd.DataFrame({'filename': train_paths, 'label': labels[train_ids - 1] - 1})
df_val = pd.DataFrame({'filename': val_paths, 'label': labels[val_ids - 1] - 1})
df_test = pd.DataFrame({'filename': test_paths, 'label': labels[test_ids - 1] - 1})

df_train['label'] = df_train['label'].astype(str)
df_val['label'] = df_val['label'].astype(str)
df_test['label'] = df_test['label'].astype(str)


print(f"✅ Eğitim için {len(df_train)} görsel bulundu.")
print(f"✅ Doğrulama için {len(df_val)} görsel bulundu.")
print(f"✅ Test için {len(df_test)} görsel bulundu.")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    channel_shift_range=20.0,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

print(f"Eğitilen fotoğraf sayısı:{len(df_train)}")

batch_size = 48
train_generator = train_datagen.flow_from_dataframe(
    df_train, x_col='filename', y_col='label', target_size=(299, 299),
    batch_size=batch_size, class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    df_val, x_col='filename', y_col='label', target_size=(299, 299),
    batch_size=batch_size, class_mode='categorical'
)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

for layer in base_model.layers[:-100]:
    layer.trainable = False  
for layer in base_model.layers[-100:]:
    layer.trainable = True  


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.3)(x)  # Dropout oranı artırıldı
output = Dense(len(np.unique(labels)), activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=output)


initial_lr = 0.0003  
optimizer = Adam(learning_rate=initial_lr)


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


log_dir = "logs/flower_model_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)


history = model.fit(
    train_generator,
    steps_per_epoch=len(df_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(df_val) // batch_size,
    epochs=15,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
)

# Sonuçlar
train_acc = np.mean(history.history['accuracy'])
val_acc = np.mean(history.history['val_accuracy'])
train_loss = np.mean(history.history['loss'])
val_loss = np.mean(history.history['val_loss'])

print(f"📌 Ortalama Eğitim Doğruluğu: {train_acc:.4f}")
print(f"📌 Ortalama Doğrulama Doğruluğu: {val_acc:.4f}")
print(f"📌 Ortalama Eğitim Kaybı: {train_loss:.4f}")
print(f"📌 Ortalama Doğrulama Kaybı: {val_loss:.4f}")


test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    df_test, x_col='filename', y_col='label', target_size=(299, 299),
    batch_size=batch_size, class_mode='categorical', shuffle=False, drop_remainder=False
)


test_steps = math.ceil(len(df_test) / batch_size)
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"📌 Test Accuracy: {test_acc:.4f}")
print(f"📌 Test Loss: {test_loss:.4f}")


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_save/flower102_inceptionv3_{timestamp}.keras')

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, steps=test_steps, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"Gerçek etiket sayısı: {len(y_true)}")
print(f"Model tahmin sayısı: {len(y_pred)}")
print(f"Test setindeki toplam görsel: {len(df_test)}")

if len(y_true) != len(y_pred):
    raise ValueError(f"Hata! y_true ({len(y_true)}) ve y_pred ({len(y_pred)}) boyutları eşleşmiyor.")


cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')


TP = np.diag(cm)  
FP = cm.sum(axis=0) - TP  
FN = cm.sum(axis=1) - TP  
TN = cm.sum() - (TP + FP + FN)  


print("----------------------------------------------------------")
print("✅ Karışıklık Matrisi:\n", cm)
print(f"📌 Accuracy (Doğruluk): {accuracy:.4f}")
print(f"📌 Precision (Kesinlik): {precision:.4f}")
print(f"📌 Recall (Duyarlılık): {recall:.4f}")
print(f"📌 F1 Score: {f1:.4f}")
print(f"📌 True Positives (TP): {TP.sum()}")
print(f"📌 True Negatives (TN): {TN.sum()}")
print(f"📌 False Positives (FP): {FP.sum()}")
print(f"📌 False Negatives (FN): {FN.sum()}")



# --- GÖRSELLEŞTİRMELER ---

# Eğitim ve doğrulama doğruluğu / kaybı grafikleri
def plot_training_history(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(14, 5))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Eğitim Doğruluğu')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-', label='Eğitim Kaybı')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.show()

plot_training_history(history)


# Karışıklık matrisi görselleştirme
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karışıklık Matrisi')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

# Eğer elinde sınıf isimleri varsa buraya koyabilirsin, yoksa sayısal etiket olarak bırakıyoruz:
class_names = [str(i) for i in range(len(np.unique(labels)))]

plot_confusion_matrix(cm, class_names)

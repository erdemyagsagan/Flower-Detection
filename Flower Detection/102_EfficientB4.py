import numpy as np
import os
import scipy.io
import math
import pandas as pd
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tf_keras.applications import EfficientNetB4
from tf_keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 📂 **Dosya Yolları**
IMG_PATH = 'data/flower102/jpg/'
LABELS_PATH = 'data/flower102/imagelabels.mat'
SETID_PATH = 'data/flower102/setid.mat'

# 🎯 **Etiketleri yükle**
labels_data = scipy.io.loadmat(LABELS_PATH)
labels = labels_data['labels'].flatten()

# 📝 **Veri bölmelerini yükle**
setid_data = scipy.io.loadmat(SETID_PATH)
train_ids = setid_data['trnid'].flatten()
val_ids = setid_data['valid'].flatten()
test_ids = setid_data['tstid'].flatten()

# 📌 **Segmentasyon görsellerinin yolunu oluştur**
def get_segmentation_paths(ids):
    paths = [os.path.join(IMG_PATH, f"image_{i:05d}.jpg") for i in ids]
    return [p for p in paths if os.path.exists(p)]

train_paths = get_segmentation_paths(train_ids)
val_paths = get_segmentation_paths(val_ids)
test_paths = get_segmentation_paths(test_ids)

df_val = pd.DataFrame({'filename': train_paths, 'label': labels[train_ids - 1] - 1})
df_test = pd.DataFrame({'filename': val_paths, 'label': labels[val_ids - 1] - 1})
df_train = pd.DataFrame({'filename': test_paths, 'label': labels[test_ids - 1] - 1})

# 🛠️ **Label'ları string'e çevir**
df_train['label'] = df_train['label'].astype(str)
df_val['label'] = df_val['label'].astype(str)
df_test['label'] = df_test['label'].astype(str)

# ✅ **Veri artırma (Augmentation)**
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

batch_size = 48
img_size = (380, 380)  # EfficientNetB4'ün giriş boyutu

train_generator = train_datagen.flow_from_dataframe(
    df_train, x_col='filename', y_col='label', target_size=img_size,
    batch_size=batch_size, class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    df_val, x_col='filename', y_col='label', target_size=img_size,
    batch_size=batch_size, class_mode='categorical'
)

# ✅ **EfficientNetB4 Modeli**
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))

# **İlk 450 katmanı dondur, son 50 katmanı eğitime aç**
for layer in base_model.layers[:450]:
    layer.trainable = False  
for layer in base_model.layers[450:]:
    layer.trainable = True  

# **Yeni katmanlar ekle**
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.4)(x)  # EfficientNet için dropout artırıldı
output = Dense(len(np.unique(labels)), activation='softmax')(x)

# **Modeli oluştur**
model = Model(inputs=base_model.input, outputs=output)

# **Öğrenme oranı ayarı**
initial_lr = 0.0002  
optimizer = Adam(learning_rate=initial_lr)

# **Öğrenme oranını dinamik olarak azalt**
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# **Erken durdurma**
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# **TensorBoard için log oluştur**
log_dir = "logs/flower_model_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# **Modeli derle**
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# 🏋️ **Modeli eğit**
history = model.fit(
    train_generator,
    steps_per_epoch=len(df_train) // batch_size,
    validation_data=val_generator,
    validation_steps=len(df_val) // batch_size,
    epochs=30,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
)

# ✅ **Sonuçları yazdır**
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    df_test, x_col='filename', y_col='label', target_size=img_size,
    batch_size=batch_size, class_mode='categorical', shuffle=False
)

test_steps = math.ceil(len(df_test) / batch_size)
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"📌 Test Accuracy: {test_acc:.4f}")
print(f"📌 Test Loss: {test_loss:.4f}")


# ✅ **Modeli kaydet**
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_save/flower102_efficientb4_{timestamp}.keras')

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




import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# 📂 Dosya yolları
TFLITE_MODEL_PATH = "model_save/resnet152_erdem2.tflite"
IMAGE_PATH = "img/antoryum.jpeg"
CLASS_CSV_PATH = "data/flower102/oxford_flower_102_name.csv"

# 🔍 Sınıf isimlerini yükle
class_names_df = pd.read_csv(CLASS_CSV_PATH)
class_names = class_names_df['Name'].tolist()

# 📸 Görseli yükle ve ön işle
img = Image.open(IMAGE_PATH).resize((224, 224))  # Model boyutuna göre ayarla
img_array = np.array(img).astype(np.float32) / 255.0
input_data = np.expand_dims(img_array, axis=0)

# 🔍 TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# 📌 Giriş ve çıkış tensorlarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🎯 Modele veriyi ver ve çalıştır
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# 📤 Çıktıyı al
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = output_data[0]

# 🧠 En yüksek olasılığa sahip sınıfı bul
top_3_indices = np.argsort(predictions)[::-1][:3]
top_3_probs = predictions[top_3_indices]

# 🖨️ Sonuçları yazdır
print("En yüksek tahmin edilen 3 çiçek:")
for i, idx in enumerate(top_3_indices):
    print(f"{i+1}. {class_names[idx]} - {top_3_probs[i]*100:.2f}%")

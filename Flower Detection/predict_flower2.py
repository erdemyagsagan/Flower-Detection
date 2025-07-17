import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tf_keras.applications.resnet import preprocess_input

# 🧼 Uyarıları sustur
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 📁 TFLite model dosyası
TFLITE_MODEL_PATH = "model_save/resnet152_erdem2.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 🔠 Oxford 102 Class İsimleri
class_names = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"
]

# Görseli modele uygun hale getir
def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0).astype(np.float32)

# Tek görseli tahmin et
def evaluate_single_image(img_path):
    img_arr = prepare_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    predicted_name = class_names[predicted_class]
    predicted_probability = output[0][predicted_class] * 100

    print(f"\n📸 Görsel: {os.path.basename(img_path)}")
    print(f"🔍 Tahmin: {predicted_name} ({predicted_probability:.2f}%)")

# ➕ Örnek kullanım
evaluate_single_image("img/daisy22.jpg")
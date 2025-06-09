import os
import pandas as pd
from ultralytics import YOLO

# === CONFIGURA QUI ===
MODEL_PATH = r"C:\Users\simo5\Desktop\yoloprovalocale\yolo_lightning\yolov9s_flir_100epochs_aug_10v4_11\weights\best.pt"
IMAGE_DIR = r"C:\Users\simo5\Desktop\yoloprovalocale\flirdataset\test\images"
CONF_THRESHOLD = 0.25
IMG_SIZE = 640
SAVE_DIR = r"runs/predict/test_results"  
CSV_PATH = os.path.join(SAVE_DIR, "predictions_report.csv")

# === CARICAMENTO MODELLO ===
model = YOLO(MODEL_PATH)

# === INFERENZA ===
results = model.predict(
    source=IMAGE_DIR,
    conf=CONF_THRESHOLD,
    imgsz=IMG_SIZE,
    save=True,
    save_txt=True,
    save_crop=True,
    project="runs/predict",
    name="test_results",
    exist_ok=True
)

# === ESTRAZIONE DATI IN CSV ===
predictions = []

for result in results:
    im_path = result.path
    boxes = result.boxes
    if boxes is None: continue

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        predictions.append({
            "image": os.path.basename(im_path),
            "class_id": cls_id,
            "confidence": round(conf, 3),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1)
        })

# === SALVA CSV ===
os.makedirs(SAVE_DIR, exist_ok=True)
df = pd.DataFrame(predictions)
df.to_csv(CSV_PATH, index=False)
print(f"\n✅ Predizioni salvate in {CSV_PATH}")
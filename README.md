
# YOLO on FLIR ADAS Thermal Dataset 🔥📷

Il dataset utilizzato è il **Teledyne FLIR ADAS Thermal Dataset (versione 2)**.

## 🧠 Obiettivi del progetto

- Allenare modelli **YOLOv8–YOLOv12** su immagini termiche.  
- Confrontare prestazioni tra modelli **one-stage (YOLO)** e **two-stage (RF-DETR)**.  
- Testare tecniche di **data augmentation avanzate** con **Albumentations**, incluse:
  - Preset 9, 10, 11  
  - Estensioni preset 10, 11
  - Combinazioni multi-preset 

## 📦 Dataset

- Dataset: [FLIR ADAS Thermal v2](https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2)  
- Formato: convertito da COCO a YOLO con `coco_to_yolo.py`  
- Split: `train`, `val`, `test`  
- Annotazioni: verificate anche con [Roboflow](https://roboflow.com)

## 🔧 Requisiti

```bash
pip install -r requirements.txt
```

Oppure installa manualmente:

```bash
pip install ultralytics wandb albumentations opencv-python
```

## 🚀 Training

### YOLOv9s con W&B logging

```bash
python scripts/train_yolo_wandb.py
```

Modifica i parametri direttamente nello script o passali da terminale (modello, dataset, epochs, ecc.).

### RF-DETR

```bash
python scripts/train_rfdetr_wandb.py
```


## 🧪 Validazione e analisi metriche

Per generare i CSV e i riassunti metrici:

```bash
python validations/create_summary_all.py
python validations/create_map5095_class_csv.py
```

Output:
- `summary_all.csv`  
- `map5095_per_class.csv`

## 🧱 Augmentazione

### Dataset aumentato con Albumentations:

- Script singolo:
  ```bash
  python scripts/augment_flir_dataset.py
  ```

- Multi-preset:
  ```bash
  python scripts/augment_flir_dataset_multi_preset.py
  ```


##  Risultati

Sono disponibili:
- CSV per ogni modello contenente le varie metriche (class,images,instances,precision,recall,map50,map50-95) per ogni categoria
- CSV summary su tutte le classi
- CSV di confronto per classe sia per le versioni small che per le augmentations

## Link utili

- 📂 **Drive con i risultati del training:**  
  

- 📊 **Tracking esperimenti con W&B:**  
  https://wandb.ai/simone-petrosino

## 👨Tesi

Questo progetto è parte della mia tesi di laurea triennale presso l'Università di Firenze, incentrata sull’uso di tecniche di deep learning per il riconoscimento di oggetti in ambienti a bassa visibilità tramite immagini termiche.

## ✍️ Autore

**Simone Petrosino**  
GitHub: https://github.com/simonepetrosino  
W&B: https://wandb.ai/simone-petrosino

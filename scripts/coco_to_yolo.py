import json
import os

# Percorsi assoluti ai file COCO e alle directory di output
coco_train_file = ""  # File COCO per il train
coco_val_file = ""      # File COCO per il validation
output_dir_train = ""    # Directory per le etichette YOLO del train
output_dir_val = ""        # Directory per le etichette YOLO del validation

# Funzione per convertire COCO in YOLO
def convert_coco_to_yolo(coco_annotation_file, output_dir):
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Carica il file COCO
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Mappa le categorie COCO a ID YOLO
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # Processa le annotazioni
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']  # [x_min, y_min, width, height]

        # Trova l'immagine corrispondente
        image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        image_width = image_info['width']
        image_height = image_info['height']
        image_filename = image_info['file_name']

        # Converti il bounding box in formato YOLO
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width /= image_width
        height /= image_height

        # Ottieni l'ID della classe YOLO
        yolo_class_id = categories[category_id]

        # Percorso del file di etichetta YOLO
        label_file = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.txt")

        # Crea la directory padre del file se non esiste
        os.makedirs(os.path.dirname(label_file), exist_ok=True)

        # Scrivi l'etichetta nel file
        with open(label_file, 'a') as f:
            f.write(f"{yolo_class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Conversione completata per {coco_annotation_file}!")

# Converti il file COCO del train
convert_coco_to_yolo(coco_train_file, output_dir_train)

# Converti il file COCO del validation
convert_coco_to_yolo(coco_val_file, output_dir_val)
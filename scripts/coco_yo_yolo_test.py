import json
import os

# Percorso assoluto al file COCO test e alla directory di output YOLO
coco_test_file = r"C:\Users\simo5\Desktop\yoloprovalocale\flirdataset\coco_test.json"
output_dir_test = r"C:\Users\simo5\Desktop\yoloprovalocale\flirdataset\test\labels"

def convert_coco_to_yolo(coco_annotation_file, output_dir):
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id_to_annotations.setdefault(ann['image_id'], []).append(ann)

    for img in coco_data['images']:
        image_id = img['id']
        image_width = img['width']
        image_height = img['height']
        image_filename = img['file_name']

        # Mantieni la struttura delle sottocartelle delle immagini anche per le label
        label_file = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}.txt")
        label_folder = os.path.dirname(label_file)
        os.makedirs(label_folder, exist_ok=True)

        anns = image_id_to_annotations.get(image_id, [])
        with open(label_file, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, width, height]
                x_min, y_min, width, height = bbox
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                width_norm = width / image_width
                height_norm = height / image_height
                yolo_class_id = categories[category_id]
                f.write(f"{yolo_class_id} {x_center} {y_center} {width_norm} {height_norm}\n")

    print(f"Conversione completata per {coco_annotation_file}!")

if __name__ == "__main__":
    convert_coco_to_yolo(coco_test_file, output_dir_test)
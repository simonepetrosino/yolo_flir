import os
import cv2
import albumentations as A
from glob import glob
import shutil

# Usa i tuoi preset già definiti, ad esempio:
def transforms_preset_10_v2(min_visibility):
    return A.Compose([
        A.CLAHE(p=0.4), 
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.BBoxSafeRandomCrop(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.MotionBlur(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.RandomGamma(p=0.3),
        A.Solarize(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))

def transforms_preset_11_v2(min_visibility):
    return A.Compose([
        A.Blur(blur_limit=5, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.Perspective(p=0.4),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        A.RandomGamma(p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))

def apply_double_augmentations(img_dir, label_dir, output_img_dir, output_label_dir, transform1, transform2):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    if not img_paths:
        print(f"Nessuna immagine trovata nella directory: {img_dir}")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossibile caricare l'immagine: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if not os.path.exists(label_path) or not os.path.getsize(label_path):
            print(f"File delle etichette non trovato o vuoto: {label_path}")
            continue

        bboxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height, int(class_id)])

        # Salva originale
        out_img_orig = os.path.join(output_img_dir, os.path.basename(img_path))
        out_lbl_orig = os.path.join(output_label_dir, os.path.basename(label_path))
        cv2.imwrite(out_img_orig, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(out_lbl_orig, "w") as f:
            for bbox in bboxes:
                x_center, y_center, width, height, class_id = bbox
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
        print(f"Immagine originale salvata: {out_img_orig}")
        print(f"Etichette originali salvate: {out_lbl_orig}")

        # Augmentazione 1
        t1 = transform1(image=img, bboxes=bboxes, class_labels=[bbox[4] for bbox in bboxes])
        if t1["bboxes"]:
            out_img_aug1 = os.path.join(output_img_dir, "aug1_" + os.path.basename(img_path))
            out_lbl_aug1 = os.path.join(output_label_dir, "aug1_" + os.path.basename(label_path))
            cv2.imwrite(out_img_aug1, cv2.cvtColor(t1["image"], cv2.COLOR_RGB2BGR))
            with open(out_lbl_aug1, "w") as f:
                for bbox in t1["bboxes"]:
                    x_center, y_center, width, height, class_id = bbox
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            print(f"Immagine augmentata 1 salvata: {out_img_aug1}")
            print(f"Etichette augmentate 1 salvate: {out_lbl_aug1}")

        # Augmentazione 2
        t2 = transform2(image=img, bboxes=bboxes, class_labels=[bbox[4] for bbox in bboxes])
        if t2["bboxes"]:
            out_img_aug2 = os.path.join(output_img_dir, "aug2_" + os.path.basename(img_path))
            out_lbl_aug2 = os.path.join(output_label_dir, "aug2_" + os.path.basename(label_path))
            cv2.imwrite(out_img_aug2, cv2.cvtColor(t2["image"], cv2.COLOR_RGB2BGR))
            with open(out_lbl_aug2, "w") as f:
                for bbox in t2["bboxes"]:
                    x_center, y_center, width, height, class_id = bbox
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            print(f"Immagine augmentata 2 salvata: {out_img_aug2}")
            print(f"Etichette augmentate 2 salvate: {out_lbl_aug2}")

def main():
    # Percorsi assoluti del dataset originale
    img_dir_train = ""
    label_dir_train = ""
    img_dir_val = ""
    label_dir_val = ""

    # Percorsi assoluti del dataset trasformato
    augmented_img_dir_train = ""
    augmented_label_dir_train = ""
    augmented_img_dir_val = ""
    augmented_label_dir_val = ""

    # Scegli i preset che vuoi usare
    transform1 = transforms_preset_10_v2(min_visibility=0.1)
    transform2 = transforms_preset_11_v2(min_visibility=0.1)

    # Training set: originale + due augmentazioni
    apply_double_augmentations(img_dir_train, label_dir_train, augmented_img_dir_train, augmented_label_dir_train, transform1, transform2)

    # Validation set: solo copia (senza augmentazione)
    os.makedirs(augmented_img_dir_val, exist_ok=True)
    os.makedirs(augmented_label_dir_val, exist_ok=True)
    for img_path in glob(os.path.join(img_dir_val, "*.jpg")):
        shutil.copy(img_path, os.path.join(augmented_img_dir_val, os.path.basename(img_path)))
    for label_path in glob(os.path.join(label_dir_val, "*.txt")):
        shutil.copy(label_path, os.path.join(augmented_label_dir_val, os.path.basename(label_path)))

    print("Dataset creato: originale + due augmentazioni per ogni immagine del training set.")

if __name__ == "__main__":
    main()
import os
import cv2
import albumentations as A
from glob import glob
import shutil

def transforms_preset_9(min_visibility):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.BBoxSafeRandomCrop(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))


def transforms_preset_10(min_visibility):
    return A.Compose([
        A.CLAHE(p=0.5),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.BBoxSafeRandomCrop(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))


def transforms_preset_11(min_visibility):
    return A.Compose([
        A.RGBShift(p=0.6),
        A.Blur(p=0.4),
        A.GaussNoise(p=0.4),
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
        A.Equalize(p=0.4),
        A.Perspective(p=0.4),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))

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

def transforms_preset_10_v3(min_visibility):
    return A.Compose([
        A.CLAHE(p=0.6),
        A.RandomBrightnessContrast(p=0.6),
        A.Equalize(mode='cv', p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4), 
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.BBoxSafeRandomCrop(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3), 
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))

def transforms_preset_10_v4(min_visibility):
    return A.Compose([
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, border_mode=0, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.Equalize(mode='cv', p=0.5),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.BBoxSafeRandomCrop(p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=min_visibility))







def apply_augmentations(img_dir, label_dir, output_img_dir, output_label_dir, transform):
    # Crea le directory di output
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Ottieni tutte le immagini
    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    if not img_paths:
        print(f"Nessuna immagine trovata nella directory: {img_dir}")
        return

    for img_path in img_paths:
        # Carica l'immagine
        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossibile caricare l'immagine: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Carica le etichette
        label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            print(f"File delle etichette non trovato: {label_path}")
            continue
        if not os.path.getsize(label_path):
            print(f"File delle etichette vuoto: {label_path}")
            continue

        bboxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height, int(class_id)])

        # Salva l'immagine originale e le etichette
        output_img_path_original = os.path.join(output_img_dir, os.path.basename(img_path))
        output_label_path_original = os.path.join(output_label_dir, os.path.basename(label_path))
        cv2.imwrite(output_img_path_original, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(output_label_path_original, "w") as f:
            for bbox in bboxes:
                x_center, y_center, width, height, class_id = bbox
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
        print(f"Immagine originale salvata: {output_img_path_original}")
        print(f"Etichette originali salvate: {output_label_path_original}")

        # Applica le augmentations
        transformed = transform(image=img, bboxes=bboxes, class_labels=[bbox[4] for bbox in bboxes])
        if not transformed["bboxes"]:
            print(f"Nessuna bounding box valida dopo la trasformazione per l'immagine: {img_path}")
            continue

        augmented_img = transformed["image"]
        augmented_bboxes = transformed["bboxes"]

        # Salva l'immagine trasformata
        output_img_path_augmented = os.path.join(output_img_dir, "aug_" + os.path.basename(img_path))
        cv2.imwrite(output_img_path_augmented, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
        print(f"Immagine augmentata salvata: {output_img_path_augmented}")

        # Salva le etichette trasformate
        output_label_path_augmented = os.path.join(output_label_dir, "aug_" + os.path.basename(label_path))
        with open(output_label_path_augmented, "w") as f:
            for bbox in augmented_bboxes:
                x_center, y_center, width, height, class_id = bbox
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
        print(f"Etichette augmentate salvate: {output_label_path_augmented}")


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

    # Scegli il preset di Albumentations
    transform = transforms_preset_10_v4(min_visibility=0.1)

    # Applica le augmentations al dataset di training
    apply_augmentations(img_dir_train, label_dir_train, augmented_img_dir_train, augmented_label_dir_train, transform)

    # Copia il validation set senza modifiche
    if not os.path.exists(augmented_img_dir_val):
        os.makedirs(augmented_img_dir_val)
    if not os.path.exists(augmented_label_dir_val):
        os.makedirs(augmented_label_dir_val)

    for img_path in glob(os.path.join(img_dir_val, "*.jpg")):
        # Copia le immagini del validation set
        output_img_path = os.path.join(augmented_img_dir_val, os.path.basename(img_path))
        shutil.copy(img_path, output_img_path)
        print(f"Immagine di validazione copiata: {output_img_path}")

    for label_path in glob(os.path.join(label_dir_val, "*.txt")):
        # Copia le etichette del validation set
        output_label_path = os.path.join(augmented_label_dir_val, os.path.basename(label_path))
        shutil.copy(label_path, output_label_path)
        print(f"Etichetta di validazione copiata: {output_label_path}")

    print("Augmentations applicate al training set e validation set copiato senza modifiche!")


if __name__ == "__main__":
    main()
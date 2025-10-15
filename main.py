# import os
# from datasets import load_dataset
# import shutil

# base_dir = "./images"
# train_dir = os.path.join(base_dir, "train")
# valid_dir = os.path.join(base_dir, "valid")

# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(valid_dir, exist_ok=True)

# dataset = load_dataset("abinazmi/rugby-player-detection-v2")

# print("Splits disponibles:", dataset.keys())

# def save_split(split_name, split_dir):
#     for idx, item in enumerate(dataset[split_name]):
#         # Imagen
#         img_name = f"{split_name}_{idx}.jpg"
#         img_path = os.path.join(split_dir, img_name)
#         shutil.copy(item['image'].filename, img_path)

#         # Label
#         label_txt = os.path.splitext(img_path)[0] + ".txt"
#         with open(label_txt, "w") as f:
#             for box in item['label']:
#                 class_id = box['class_id']
#                 x_center = box['x_center']
#                 y_center = box['y_center']
#                 width = box['width']
#                 height = box['height']
#                 f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# if 'train' in dataset:
#     print("Descargando y guardando train set...")
#     save_split('train', train_dir)
# if 'valid' in dataset:
#     print("Descargando y guardando valid set...")
#     save_split('valid', valid_dir)

# print("✅ Dataset v2 listo para YOLOv8")
import os
import glob
import shutil

dataset_path = "C:\\rugby"  
splits = ["train", "valid"]


def fix_labels(split):
    label_dir = os.path.join(dataset_path, split, "labels")
    img_dir = os.path.join(dataset_path, split, "images")

    print(f"\n🔹 Revisando {split}...")
    img_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{img_dir}/*")}
    label_files = glob.glob(f"{label_dir}/*.txt")

    corrupt_count = 0
    fixed_count = 0

    for file in label_files:
        base = os.path.splitext(os.path.basename(file))[0]
        if base not in img_files:
            print(f" Label sin imagen: {file}")
            corrupt_count += 1
            continue

        with open(file, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = "0" 
                new_lines.append(" ".join(parts))
            else:
                corrupt_count += 1
        if new_lines != lines:
            with open(file, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            fixed_count += 1

    print(f"✅ {split}: {len(label_files)} labels procesadas, {fixed_count} corregidas, {corrupt_count} corruptas.")

def clear_cache(split):
    cache_file = os.path.join(dataset_path, split, "labels.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"🗑️  Cache eliminado: {cache_file}")

for split in splits:
    fix_labels(split)
    clear_cache(split)

print("\n🎉 Dataset listo para YOLOv8.")

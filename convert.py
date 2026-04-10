from PIL import Image
import pillow_heif
import os

pillow_heif.register_heif_opener()

sub_folder = 'test'

dataset_path = f'./data/gallery-heic/{sub_folder}'
output_path = f'./data/gallery/{sub_folder}'
files = os.listdir(dataset_path)

os.makedirs(output_path, exist_ok=True)
counter = 1

for file in files:
    ext = os.path.splitext(file)[1].lower()
    if ext != ".heic":
        # copy over to output folder if it's not a heic file
        print(f"{counter}: Copying {file} to output folder...")
        with Image.open(os.path.join(dataset_path, file)) as img:
            img.save(os.path.join(output_path, file))
    else:
        print(f"{counter}: Converting {file} to JPEG...")
        with Image.open(os.path.join(dataset_path, file)) as img:
            jpg_name = f"{os.path.splitext(file)[0]}.jpg"
            jpg_path = os.path.join(output_path, jpg_name)
            img.convert("RGB").save(jpg_path, format="JPEG", quality=95)
    counter += 1

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
import h5py
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter
import glob

# Constants from AGCHDataModule
IMAGE_DIM = 4096
TEXT_DIM = 1386
NUM_CLASSES = 24
DATA_DIR = Path("data")

URL_IMAGES = "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip"
URL_ANNOTATIONS = (
    "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip"
)


def download_url(url, output_path):
    if output_path.exists():
        print(f"File {output_path} already exists, skipping download.")
        return

    print(f"Downloading {url} to {output_path}...")
    # Add fake user agent to avoid 403
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, output_path)


def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def get_vgg16_extractor():
    print("Loading VGG16 model...")
    # Use VGG16, take features from fc7 (second to last fc layer usually)
    # VGG16 structure: features -> avgpool -> classifier
    # classifier: (0): Linear(25088, 4096) -> (1): ReLU -> (2): Dropout
    #          -> (3): Linear(4096, 4096) -> (4): ReLU -> (5): Dropout -> (6): Linear(4096, 1000)
    # We want output of (4) or input to (6). Usually fc7 is the second 4096 layer.
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Remove the last classification layer (6)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def process_images(image_dir, output_file):
    image_paths = sorted(glob.glob(str(image_dir / "*.jpg")))
    num_images = len(image_paths)
    print(f"Found {num_images} images.")

    # Check simple limit for test
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        image_paths = image_paths[:50]
        num_images = 50
        print("Test mode: processing 50 images only.")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = get_vgg16_extractor()
    device = next(model.parameters()).device

    features = np.zeros((num_images, IMAGE_DIM), dtype=np.float32)

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, num_images, batch_size), desc="Extracting Visual Features"):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = []
            valid_indices = []

            for idx, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    img_t = preprocess(img)
                    batch_imgs.append(img_t)
                    valid_indices.append(i + idx)
                except Exception as e:
                    print(f"Error reading {p}: {e}")

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(device)
            # Output of VGG fc7 is 4096
            feats = model(batch_tensor)
            features[valid_indices] = feats.cpu().numpy()

    return features, image_paths


def process_text_and_labels(image_paths, text_dir, annotation_dir, num_images):
    print("Processing tags and labels...")

    # 1. Load all tags
    # MIRFlickr tags are usually in 'mirflickr25k/mirflickr/meta/tags_r1.txt' or similar per image?
    # Actually, raw download structure: mirflickr25k/mirflickr/im1.jpg ... and meta/tags/im1.txt

    # Let's check structure after download, usually:
    # mirflickr/
    #   im1.jpg ...
    #   meta/
    #     tags/
    #       im1.txt

    # We need to build vocabulary from top TEXT_DIM tags
    all_tags_list = []
    img_tags_map = {}  # path_basename -> [tags]

    for p in image_paths:
        basename = Path(p).stem  # im1
        # Tag file
        tag_file = Path(p).parent / "meta" / "tags" / (basename + ".txt")
        if not tag_file.exists():
            # Try alternate structure if extracted differently
            # The zip usually contains "mirflickr" folder
            # If we extracted to data/mirflickr, then path matches
            continue

        with open(tag_file, "r", encoding="utf-8", errors="ignore") as f:
            tags = [line.strip().lower() for line in f if line.strip()]
            img_tags_map[basename] = tags
            all_tags_list.extend(tags)

    # Build vocab
    counts = Counter(all_tags_list)
    # Top 1386 tags
    vocab = [tag for tag, _ in counts.most_common(TEXT_DIM)]
    vocab_map = {tag: i for i, tag in enumerate(vocab)}

    # Build text features (BoW or Multi-hot)
    text_features = np.zeros((num_images, TEXT_DIM), dtype=np.float32)

    for i, p in enumerate(image_paths):
        basename = Path(p).stem
        tags = img_tags_map.get(basename, [])
        for tag in tags:
            if tag in vocab_map:
                text_features[i, vocab_map[tag]] = 1.0

    # 2. Process Labels
    # Annotations zip structure: mirflickr25k_annotations_v080/ Usually .txt files per class
    # e.g., clouds.txt containing indices (1-based)

    # Load annotation files
    lbl_files = sorted(glob.glob(str(annotation_dir / "*.txt")))
    # We need exactly 24 classes. Filter strict standard if needed?
    # Standard 24 classes:
    standard_labels = [
        "animals",
        "baby",
        "bird",
        "car",
        "clouds",
        "dog",
        "female",
        "flower",
        "food",
        "indoor",
        "lake",
        "male",
        "night",
        "people",
        "plant_life",
        "portrait",
        "river",
        "sea",
        "sky",
        "structures",
        "sunset",
        "transport",
        "tree",
        "water",
    ]

    labels = np.zeros((num_images, NUM_CLASSES), dtype=np.float32)

    # Map from image number to index in our features array
    # Our features array is sorted by filename (im1.jpg, im10.jpg...) ?
    # Wait, glob sorted order might be im1.jpg, im10.jpg ...
    # MIRFlickr IDs are usually just the number.

    # Let's build a map: 'im1' -> 0, 'im2' -> ...
    name_to_idx = {Path(p).stem: i for i, p in enumerate(image_paths)}

    for i, lbl_name in enumerate(standard_labels):
        # Look for lbl_name.txt or similar
        # Some might have _r1.txt (relevant). We usually take the broad one? Or strict?
        # Standard is usually broad.
        fpath = annotation_dir / (lbl_name + ".txt")
        if not fpath.exists():
            print(f"Warning: Label file {lbl_name}.txt not found.")
            continue

        with open(fpath, "r") as f:
            for line in f:
                try:
                    # File lines are image numbers (e.g. 1, 25, ...)
                    img_id = f"im{line.strip()}"
                    if img_id in name_to_idx:
                        labels[name_to_idx[img_id], i] = 1.0
                except:
                    pass

    return text_features, labels


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Download
    zip_img = DATA_DIR / "mirflickr25k.zip"
    zip_anno = DATA_DIR / "mirflickr_annotations.zip"

    download_url(URL_IMAGES, zip_img)
    download_url(URL_ANNOTATIONS, zip_anno)

    # 2. Extract
    extract_img_dir = DATA_DIR / "mirflickr"  # Extraction usually creates a subdir
    if not extract_img_dir.exists():
        extract_zip(zip_img, DATA_DIR)

    extract_anno_dir = DATA_DIR / "mirflickr25k_annotations_v080"
    if not extract_anno_dir.exists():
        extract_zip(zip_anno, DATA_DIR)

    # Verify paths after extraction
    # The zip `mirflickr25k.zip` usually contains a folder `mirflickr`
    if not (DATA_DIR / "mirflickr").exists():
        # Maybe it extracted flat?
        print("Warning: Expected 'mirflickr' folder not found. Checking structure...")

    img_folder = DATA_DIR / "mirflickr"
    anno_folder = DATA_DIR  # Annotations extracted flat into DATA_DIR

    # 3. Process
    # Ensure raw images are in img_folder
    # Sometimes structure is mirflickr/im1.jpg

    print("Processing Visual Features...")
    features, image_paths = process_images(img_folder, DATA_DIR / "images_raw.npy")

    print("Processing Text/Labels...")
    texts, labels = process_text_and_labels(image_paths, img_folder, anno_folder, len(image_paths))

    print(f"Shapes: Images {features.shape}, Texts {texts.shape}, Labels {labels.shape}")

    # 4. Save to HDF5
    print("Saving HDF5...")
    with h5py.File(DATA_DIR / "images.h5", "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("labels", data=labels)

    with h5py.File(DATA_DIR / "texts.h5", "w") as f:
        f.create_dataset("features", data=texts)
        f.create_dataset("labels", data=labels)  # redundant but sometimes useful

    print("Done! Data prepared in data/")


if __name__ == "__main__":
    main()

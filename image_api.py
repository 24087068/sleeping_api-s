import os
import shutil

import requests
import pandas as pd
import time
class ImageAPI:
    def __init__(self, api_url, headers, save_folder="data/images"):
        self.api_url = api_url
        self.headers = headers
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True) # saves images in data/images

    def search_images(self, query, limit=3):
        """Search Wikimedia for images titles matching model names"""
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": 6,
            "srlimit": limit
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        data = r.json()
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]

    # GenAI: Get all images for firearm models from the audio dataset names, https://chatgpt.com/share/68ebef06-e254-800a-b767-1649251a10ec
    def get_image_info(self, titles):
        """Get image URLs from Wikimedia titles"""
        if not titles:
            return []
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url"
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        pages = r.json().get("query", {}).get("pages", {})
        urls = []
        for p in pages.values():
            if "imageinfo" in p:
                urls.append(p["imageinfo"][0]["url"])
        return urls

    def download_image(self, url, filename):
        headers = self.headers  # reuse your real UA header
        r = requests.get(url, headers=headers, allow_redirects=True)

        # Check if we actually downloaded an image
        content_type = r.headers.get("Content-Type", "")

        if not content_type.startswith("image/"):
            print(f"Warning: URL returned non-image content: {content_type}")
            print(f"URL was: {url}")
            return False

        with open(os.path.join(self.save_folder, filename), "wb") as f:
            f.write(r.content)

        return True

    def fetch_images_for_models(self, models):
        """Run search and download for a list of models, create csv file for metadata"""
        metadata = []
        for model in models:
            print(f"Searching for {model}...")
            titles = self.search_images(model)
            urls = self.get_image_info(titles)
            if urls:
                img_url = urls[0]
                filename = f"{model.replace(' ', '_')}.jpg"
                self.download_image(img_url, filename)
                metadata.append({"model": model, "url": img_url, "path": os.path.join(self.save_folder, filename)})
                print(f"Downloaded: {img_url}")
            else:
                print(f"No image found for {model}.")
            time.sleep(1)  # avoid hitting the API too fast
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.save_folder, "image_metadata.csv"), index=False)
        return df
import os
import pandas as pd
from PIL import Image
import shutil
import logging
import matplotlib.pyplot as plt
import sqlite3

def organize_images_by_weapon(
    source_dir="data/images",
    raw_dir="data/images/raw",
    extensions=(".jpg", ".jpeg", ".png")
):
    """
    Move/copy images into subfolders per weapon model.
    data/images/AK47.jpg -> data/images/raw/AK47/AK47.jpg
    """

    os.makedirs(raw_dir, exist_ok=True)
    processed = 0

    for img_file in os.listdir(source_dir):
        if img_file.lower().endswith(extensions):
            weapon_name = os.path.splitext(img_file)[0]
            weapon_dir = os.path.join(raw_dir, weapon_name)
            os.makedirs(weapon_dir, exist_ok=True)

            src = os.path.join(source_dir, img_file)
            dst = os.path.join(weapon_dir, img_file)

            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy(src, dst)
                processed += 1

    logging.info(f"Organized {processed} images into {raw_dir}")
# image_api.py
def process_weapon_images(
    raw_base_dir="data/images/raw",
    processed_base_dir="data/images/processed"
):
    """
    Extract basic visual features per weapon image.
    """

    os.makedirs(processed_base_dir, exist_ok=True)
    records = []

    for weapon_key in os.listdir(raw_base_dir):
        weapon_dir = os.path.join(raw_base_dir, weapon_key)
        if not os.path.isdir(weapon_dir):
            continue

        for img_name in os.listdir(weapon_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(weapon_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    aspect_ratio = round(width / height, 3)

                    # Heuristic rules (same logic as notebook intent)
                    has_stock_prediction = int(aspect_ratio > 2.0)
                    handling_type = "2-handed" if has_stock_prediction else "1-handed"
                    stock_label_nl = "Met kolf" if has_stock_prediction else "Zonder kolf"

                    records.append({
                        "weapon_key": weapon_key,
                        "image_name": img_name,
                        "pixel_width": width,
                        "pixel_height": height,
                        "aspect_ratio": aspect_ratio,
                        "has_stock_prediction": has_stock_prediction,
                        "handling_type": handling_type,
                        "stock_label_nl": stock_label_nl
                    })

            except Exception as e:
                logging.warning(f"Failed processing {img_path}: {e}")

    return pd.DataFrame(records)

def visualize_image_features(processed_df):
    """
    Visualize extracted image features (EDA only).
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Handling type distribution
    handling_counts = processed_df['handling_type'].value_counts()
    axes[0, 0].bar(handling_counts.index, handling_counts.values)
    axes[0, 0].set_title('Handling Type Distribution')
    axes[0, 0].set_ylabel('Count')

    # Aspect ratio distribution
    axes[0, 1].hist(processed_df['aspect_ratio'], bins=20, edgecolor='black')
    axes[0, 1].axvline(2.0, linestyle='--', linewidth=2)
    axes[0, 1].set_title('Aspect Ratio Distribution')

    # Pixel dimensions
    axes[1, 0].scatter(
        processed_df['pixel_width'],
        processed_df['pixel_height'],
        c=processed_df['has_stock_prediction'],
        alpha=0.6
    )
    axes[1, 0].set_title('Pixel Dimensions')

    # Handling pie chart
    stock_dist = processed_df.groupby('handling_type').size()
    axes[1, 1].pie(stock_dist.values, labels=stock_dist.index, autopct='%1.1f%%')
    axes[1, 1].set_title('1-handed vs 2-handed')

    plt.tight_layout()
    plt.show()


def save_image_features_to_db(
    processed_df,
    db_path="weapons_data.db",
    table_name="image_features"
):
    """
    Save extracted image features to SQLite.
    """
    with sqlite3.connect(db_path) as conn:
        processed_df.to_sql(table_name, conn, if_exists='replace', index=False)
import os
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

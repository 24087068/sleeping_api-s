import requests

def get_image_api(model, api_url, headers):
    # 1️⃣ Search for the page about the firearm
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": model,
        "srlimit": 1
    }
    r = requests.get(api_url, params=params, headers=headers)
    search_data = r.json()
    results = search_data.get("query", {}).get("search", [])
    if not results:
        return None

    page_title = results[0]["title"]

    # 2️⃣ Get images on that page
    params2 = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "images"
    }
    r2 = requests.get(api_url, params=params2, headers=headers)
    images_data = r2.json()
    pages = images_data.get("query", {}).get("pages", {})
    if not pages:
        return None

    images = list(pages.values())[0].get("images", [])
    if not images:
        return None

    # 3️⃣ Get full URL of the first image
    image_title = images[0]["title"]
    params3 = {
        "action": "query",
        "format": "json",
        "titles": image_title,
        "prop": "imageinfo",
        "iiprop": "url"
    }
    r3 = requests.get(api_url, params=params3, headers=headers)
    imageinfo_data = r3.json()
    pages = imageinfo_data.get("query", {}).get("pages", {})
    if not pages:
        return None

    imginfo = list(pages.values())[0].get("imageinfo", [])
    if not imginfo:
        return None

    return imginfo[0]["url"]

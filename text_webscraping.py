from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import quote
import re

# test functie with texts from web scraping
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
def get_fandom_text(url):
    response = requests.get(url, headers={"User-Agent": "Pipline/1.0 (mkakol.index@gmail.com)"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.find_all("p")
    return text

# get all texts from fandom
# GenAi: Using Special search query in web scraping, https://chatgpt.com/share/695ace2a-4144-800a-b47e-d1787e6ce53c
def get_fandom_texts(audio_df):
    # Base urls
    base_url = "https://guns.fandom.com/wiki/"
    search_url = base_url + "Special:Search?query="
    results = []
    # Loop over all firearm models from audio_df
    for model in audio_df["model"]:
        # Replace - with spaces to improve results
        search_term = model.replace("-", " ")
        try:
            # Search for the firearm page
            response = requests.get(search_url + quote(search_term), timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            # Take the first search result
            first_result = soup.find("a", class_="unified-search__result__title")
            if not first_result or not first_result.get("href"):
                # No page found for this model
                results.append({
                    "model": model,
                    "description_text": "Null"
                })
                continue
            firearm_url = first_result["href"]
            # Open the firearm page itself
            page = requests.get(firearm_url, timeout=5)
            page_soup = BeautifulSoup(page.text, "html.parser")
            # Main article content is inside this div
            article_body = page_soup.find("div", class_="mw-parser-output")
            description = "Not found"
            if article_body:
                # First non-empty paragraph is usually the description
                for p in article_body.find_all("p"):
                    text = p.get_text(strip=True)
                    if text:
                        description = text
                        break
            results.append({
                "model": model,
                "description_text": description
            })
        except requests.RequestException:
            # Handles cases where the page cannot be reached
            results.append({
                "model": model,
                "description_text": "Not found"
            })
    return pd.DataFrame(results)

def clean_weapon_data(df):
    def extracts(text):
        if not text or text in ["Null", "Not found"]:
            return "Not found", "Not found"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
        # GenAi, Getting filtering spaces right: https://chatgpt.com/share/69737cb1-59a8-800a-b37a-25b53033de4e
        type_match = re.search(r'weapon\s*type\s*:?\s*([A-Za-z0-9\s\-\(\),\.]+?)(?=\s*caliber|\s*action|\s*length|$)', text, re.IGNORECASE)
        w_type = type_match.group(1).strip() if type_match else "Not found"
        cal_match = re.search(r'caliber\s*:?\s*([\d\.\w\s\-x/]+)', text, re.IGNORECASE)
        caliber = cal_match.group(1).strip() if cal_match else ""
        caliber = re.split(r'\s{2,}|(?=[A-Z][a-z])', caliber)[0].strip()
        return w_type, caliber
    df[['type', 'caliber']] = df['description_text'].apply(lambda x: pd.Series(extracts(x)))
    return df[['model', 'type', 'caliber']]

def text_eda(df):
    print("Top 10 Types")
    print(df['type'].value_counts().head(10))
    print("Top 10 Calibers")
    print(df['caliber'].value_counts().head(10))

def simplify_types(df):
    def map_category(t):
        t = str(t).lower()
        if 'pistol' in t or 'revolver' in t or 'Semi-automatic Pistol(most models)Select-firemachine pistol(Glock 18)' in t:
            return 'pistol'
        if 'rifle' in t or 'smg' in t or 'submachine' in t or 'machine gun' in t:
            return 'rifle'
        if 'shotgun' in t:
            return 'shotgun'
        return 'other'
    text_features = df.copy()
    text_features['type'] = df['type'].apply(map_category)
    return text_features[['model', 'type', 'caliber']]
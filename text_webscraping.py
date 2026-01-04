from bs4 import BeautifulSoup
import requests
import pandas as pd
from urllib.parse import quote
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


from bs4 import BeautifulSoup
import requests
# test functie with texts from web scraping
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
def get_fandom_text(url):
    response = requests.get(url, headers={"User-Agent": "StudentProject/1.0 (mkakol.index@gmail.com)"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.find_all("p")
    return text

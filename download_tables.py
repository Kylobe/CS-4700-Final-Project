import requests
from bs4 import BeautifulSoup
import os

URL = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/"
os.makedirs("syzygy", exist_ok=True)

html = requests.get(URL).text
soup = BeautifulSoup(html, "html.parser")

links = [a["href"] for a in soup.find_all("a") if a["href"].endswith(".rtbw")]

for link in links:
    fname = os.path.join("syzygy", link)
    print("Downloading:", link)
    with requests.get(URL + link, stream=True) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
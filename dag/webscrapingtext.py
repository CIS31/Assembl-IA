import requests
from bs4 import BeautifulSoup
import re
import os
from datetime import datetime
from pathlib import Path

base_url = "https://www.assemblee-nationale.fr"
url = base_url + "/dyn/17/comptes-rendus/seance"

response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

def french_date_to_datetime(date_str):
    mois_fr = {
        'janvier':1, 'février':2, 'fevrier':2, 'mars':3, 'avril':4,
        'mai':5, 'juin':6, 'juillet':7, 'août':8, 'aout':8,
        'septembre':9, 'octobre':10, 'novembre':11, 'décembre':12, 'decembre':12
    }
    parts = date_str.split('-')
    if len(parts) == 4:
        jour = int(parts[1])
        mois = mois_fr.get(parts[2].lower(), 0)
        annee = int(parts[3])
        if mois != 0:
            return datetime(annee, mois, jour)
    return None

links_dates = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if href.startswith("/dyn/17/comptes-rendus/seance/"):
        match = re.search(r"/([^/]+)$", href)
        if match:
            last_part = match.group(1)
            date_match = re.search(r"du-([^-]+-\d{1,2}-[a-zéû]+-\d{4})", last_part)
            if date_match:
                date_str = date_match.group(1)
                date_obj = french_date_to_datetime(date_str)
                if date_obj:
                    links_dates.append((href, date_str, date_obj))

if not links_dates:
    print("Aucun lien de séance trouvé sur la page.")
    exit(1)

links_dates.sort(key=lambda x: x[2], reverse=True)

unique_dates = []
seen = set()
for _, d_str, _ in links_dates:
    if d_str not in seen:
        unique_dates.append(d_str)
        seen.add(d_str)

if len(unique_dates) < 2:
    print("Moins de deux dates uniques trouvées, impossible de prendre la deuxième.")
    exit(1)

second_date_str = unique_dates[1]
print("Deuxième date la plus récente :", second_date_str)

matching_links = set()
for href, d_str, _ in links_dates:
    if d_str == second_date_str:
        href_clean = href.split("#")[0]
        if href_clean.endswith(".pdf"):
            href_clean = href_clean[:-4]
        matching_links.add(href_clean)

if not matching_links:
    print("Aucun lien trouvé pour la deuxième date extraite.")
    exit(1)

last_link = sorted(matching_links)[-1]
print("Dernier lien pour la deuxième date :", base_url + last_link)

response2 = requests.get(base_url + last_link)
soup2 = BeautifulSoup(response2.content, "html.parser")

xml_link = None
for a in soup2.find_all("a", href=True):
    href = a['href']
    if href.endswith(".xml") and "opendata" in href:
        xml_link = href
        break

if not xml_link:
    print("Lien XML non trouvé sur la page.")
    exit(1)

full_xml_url = xml_link if xml_link.startswith("http") else base_url + xml_link
print("Lien XML trouvé :", full_xml_url)

xml_response = requests.get(full_xml_url)
if xml_response.status_code == 200:
    filename = os.path.basename(full_xml_url)

    import sys
    from pathlib import Path

    try:
        from emotion_detectionvf import AzureUtils
    except ImportError:
        AzureUtils = None

    if AzureUtils is not None:
        azure_utils = AzureUtils(mount_dir="/mnt/data")
        AZURE_RUN = azure_utils.detect_azure_run()
    else:
        AZURE_RUN = False

    if AZURE_RUN:
        azure_utils.mount_dir_Azure()
        tmp_path = Path("/tmp") / filename
        with open(tmp_path, "wb") as f:
            f.write(xml_response.content)
        dest_blob = f"{azure_utils.mount_dir}/text/input/{filename}"
        dbutils.fs.mkdirs(f"{azure_utils.mount_dir}/text/input")
        dbutils.fs.cp(f"file:{tmp_path}", dest_blob, overwrite=True)
        print(f"[Blob] Fichier XML sauvegardé dans Azure → {dest_blob}")
    else:
        current_dir = Path(__file__).resolve().parent
        input_dir = current_dir.parent / "text" / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        file_path = input_dir / filename
        with open(file_path, "wb") as f:
            f.write(xml_response.content)
        print(f"[Local] Fichier XML téléchargé → {file_path}")
else:
    print(f"Erreur lors du téléchargement du fichier XML : code {xml_response.status_code}")

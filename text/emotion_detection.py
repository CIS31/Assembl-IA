#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextEmotionAnalyzer – version cloud-ready
Compatible exécution locale & Azure (DBFS / DataLake)
"""

import os
import sys
import shutil
import glob
import re
import csv                 # ← ajouté
import psycopg2            # ← ajouté
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    pipeline,
    AutoTokenizer,
)

# ───── NLTK punkt ─────
try:
    nltk.data.find("tokenizers/punkt/french.pickle")
except LookupError:
    nltk.download("punkt")

# ───── Azure Utils ─────
class AzureUtils:
    def __init__(self, mount_dir="/mnt/data"):
        self.mount_dir = mount_dir

    def detect_azure_run(self) -> bool:
        args = dict(arg.split("=") for arg in sys.argv[1:] if "=" in arg)
        return args.get("AZURE_RUN", "false").lower() == "true"

    def mount_dir_Azure(self):
        def is_mounted(mount_point):
            mounts = [m.mountPoint for m in dbutils.fs.mounts()]
            return mount_point in mounts

        configs = {
            "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-application-id"),
            "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-secret-value"),
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{dbutils.secrets.get(scope='az-kv-assemblia-scope', key='sp-tenant-id')}/oauth2/token",
        }

        if not is_mounted(self.mount_dir):
            dbutils.fs.mount(
                source="abfss://data@azbstelecomparis.dfs.core.windows.net/",
                mount_point=self.mount_dir,
                extra_configs=configs,
            )
            print(f"[AzureUtils] Montage OK → {self.mount_dir}")
        else:
            print(f"[AzureUtils] Déjà monté → {self.mount_dir}")

    def get_latest_xml(self, folder_dbfs: str) -> str:
        files = dbutils.fs.ls(folder_dbfs)
        xml_files = [f for f in files if f.name.endswith(".xml")]
        if not xml_files:
            raise FileNotFoundError(f"Aucun fichier XML trouvé dans {folder_dbfs}")
        latest = max(xml_files, key=lambda f: f.modificationTime)
        print(f"[AzureUtils] Dernier XML : {latest.path}")
        return latest.path

# ───── Postgres Utils ─────
class PostgresUtils:
    """
    Outil générique pour créer la table `textTimeline` (si besoin) et insérer les
    données du CSV généré par TextEmotionAnalyzer.
    Les paramètres de connexion sont passés en arguments du job :
        --PGHOST=… --PGDATABASE=… --PGUSER=… --PGPASSWORD=…
    """
    def __init__(self):
        args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)

        os.environ['PGHOST']     = args.get('PGHOST', '')
        os.environ['PGDATABASE'] = args.get('PGDATABASE', '')
        os.environ['PGUSER']     = args.get('PGUSER', '')
        os.environ['PGPASSWORD'] = args.get('PGPASSWORD', '')
        os.environ['PGPORT']     = args.get('PGPORT', os.getenv('PGPORT', '5432'))

        self.host     = os.getenv('PGHOST')
        self.database = os.getenv('PGDATABASE')
        self.user     = os.getenv('PGUSER')
        self.password = os.getenv('PGPASSWORD')
        self.port     = int(os.getenv('PGPORT', 5432))
        self.conn     = None

    # --------------------------------------------------------------------- connexion
    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host, database=self.database,
            user=self.user, password=self.password,
            port=self.port, sslmode="require"
        )
        print(f"[Postgres] Connexion OK → {self.database}")

    # ------------------------------------------------------------------ création table
    def create_table(self, table_name="textTimeline"):
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            docID INT,
            ordinal_prise INT,
            orateur TEXT,
            debut DOUBLE PRECISION,
            fin DOUBLE PRECISION,
            sad REAL,
            disgust REAL,
            angry REAL,
            neutral REAL,
            fear REAL,
            surprise REAL,
            happy REAL,
            texte TEXT
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(ddl)
        self.conn.commit()
        print(f"[Postgres] Table '{table_name}' vérifiée / créée.")

    # ----------------------------------------------------------- auto-incrément docID
    def get_last_doc_id(self, table_name="textTimeline"):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT MAX(docID) FROM {table_name};")
            res = cur.fetchone()[0]
        return res if res is not None else 0

    # --------------------------------------------------------------------- insertion
    def insert_csv(self, csv_path, table_name="textTimeline"):
        new_doc_id = self.get_last_doc_id(table_name) + 1
        with self.conn.cursor() as cur, open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cur.execute(
                    f"""INSERT INTO {table_name}
                        (docID, ordinal_prise, orateur, debut, fin,
                         sad, disgust, angry, neutral, fear, surprise, happy, texte)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);""",
                    (new_doc_id,
                     int(row["ordinal_prise"]), row["orateur"],
                     float(row["debut"]),  float(row["fin"]),
                     float(row["sad"]),    float(row["disgust"]),
                     float(row["angry"]),  float(row["neutral"]),
                     float(row["fear"]),   float(row["surprise"]),
                     float(row["happy"]),  row["texte"])
                )
        self.conn.commit()
        print(f"[Postgres] Insertion CSV terminée • docID={new_doc_id}")

    # ------------------------------------------------------------------- fermeture
    def close(self):
        if self.conn:
            self.conn.close()
            print("[Postgres] Connexion fermée.")

# ───── Emotion Analyzer ─────
class TextEmotionAnalyzer:
    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = model_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Model] Loading CamemBERT from: {model_dir}")
        self.model = CamembertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=5,
        )

        self.rename_map = {
            "sad": "sad",
            "disgusted": "disgust",
            "anger": "angry",
            "neutral": "neutral",
            "fear": "fear",
            "surprise": "surprise",
            "joy": "happy",
        }
        self.ordered_labels = [
            "sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"
        ]

    @staticmethod
    def _clean_text(txt: str) -> str:
        txt = txt.replace("\xa0", " ")
        return re.sub(r"\s+", " ", txt).strip()

    def _chunk_sentences(self, text: str, max_len=512):
        sentences = sent_tokenize(text, language="french")
        chunks, chunk, n = [], [], 0
        for sent in sentences:
            ids = self.tokenizer.encode(sent, add_special_tokens=False)
            if n + len(ids) + 2 > max_len:
                chunks.append(" ".join(chunk))
                chunk, n = [sent], len(ids)
            else:
                chunk.append(sent)
                n += len(ids)
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    def analyze_xml(self, xml_path: str) -> Path:
        xml_path = Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"ns": "http://schemas.assemblee-nationale.fr/referentiel"}

        prises = []
        for paragraphe in root.findall(".//ns:paragraphe[@ordinal_prise]", ns):
            texte_elem = paragraphe.find("ns:texte", ns)
            texte = "".join(texte_elem.itertext()) if texte_elem is not None else ""
            texte = self._clean_text(texte)
            if not texte:
                continue
            stime = texte_elem.attrib.get("stime") if texte_elem is not None else None
            if stime is None:
                continue
            prises.append({
                "ordinal_prise": paragraphe.attrib.get("ordinal_prise"),
                "orateur": paragraphe.find("ns:orateurs/ns:orateur/ns:nom", ns).text if paragraphe.find("ns:orateurs/ns:orateur/ns:nom", ns) is not None else "Inconnu",
                "texte": texte,
                "debut": float(stime),
            })

        if not prises:
            raise ValueError("Aucune prise de parole extraite")

        prises = sorted(prises, key=lambda x: int(x["ordinal_prise"]))
        for i in range(len(prises) - 1):
            prises[i]["fin"] = prises[i + 1]["debut"]
        prises[-1]["fin"] = None
        prises = [p for p in prises if p["fin"] is not None]

        results = []
        for p in prises:
            agg = {}
            for chunk in self._chunk_sentences(p["texte"]):
                preds = self.pipe(chunk)[0]
                for pr in preds:
                    agg[pr["label"]] = agg.get(pr["label"], 0) + pr["score"]
            total = sum(agg.values()) or 1
            agg = {k: v / total for k, v in agg.items()}
            renamed = {self.rename_map[lbl]: agg.get(lbl, 0.0) for lbl in self.rename_map}
            results.append({**p, **renamed})

        df = pd.DataFrame(results)[["ordinal_prise", "orateur", "texte", "debut", "fin"] + self.ordered_labels].fillna(0)
        out_path = self.output_dir / f"{xml_path.stem}_emotions.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] Résultats sauvegardés → {out_path}")
        return out_path

# ───── Main ─────
if __name__ == "__main__":
    azure_utils = AzureUtils(mount_dir="/mnt/data")
    AZURE_RUN = azure_utils.detect_azure_run()

    if AZURE_RUN:
        print("▶ Exécution Azure • Montage ADLS …")
        azure_utils.mount_dir_Azure()

        xml_folder_dbfs = f"{azure_utils.mount_dir}/text/input"
        model_dir_dbfs = f"{azure_utils.mount_dir}/text/models"
        output_folder_dbfs = f"{azure_utils.mount_dir}/text/output"

        latest_xml_dbfs = azure_utils.get_latest_xml(xml_folder_dbfs)
        tmp_dir = Path("/tmp/text_emotion")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        xml_local = tmp_dir / Path(latest_xml_dbfs).name
        model_local = tmp_dir / "model"
        output_local = tmp_dir / "output"
        output_local.mkdir(exist_ok=True)

        if not xml_local.exists():
            dbutils.fs.cp(latest_xml_dbfs, f"file:{xml_local}")
        if not model_local.exists():
            dbutils.fs.cp(model_dir_dbfs, f"file:{model_local}", recurse=True)

    else:
        print("▶ Exécution locale")
        xml_folder = "input"
        model_dir = "models"
        output_local = Path("output")
        output_local.mkdir(exist_ok=True)

        xml_files = sorted(Path(xml_folder).glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"Aucun XML dans {xml_folder}")
        xml_local = xml_files[-1]
        model_local = Path(model_dir)

    analyzer = TextEmotionAnalyzer(
        model_dir=str(model_local),
        output_dir=str(output_local),
    )
    csv_path = analyzer.analyze_xml(str(xml_local))

    # ---------------------------------------------------------------------- stockage
    if AZURE_RUN:
        dest_dbfs = f"{output_folder_dbfs}/{csv_path.name}"
        dbutils.fs.cp(f"file:{csv_path}", dest_dbfs)
        print("✔ Pipeline terminé dans Azure")

        # --------------------------- insertion PostgreSQL
        try:
            pg = PostgresUtils()
            pg.connect()
            pg.create_table(table_name="textTimeline")  
            pg.insert_csv(csv_path, table_name="textTimeline")
            print("✔ Données insérées dans PostgreSQL")
        finally:
            pg.close()
    else:
        print("✔ Pipeline terminé en local")

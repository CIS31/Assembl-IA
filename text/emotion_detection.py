#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextEmotionAnalyzer – version cloud-ready
Compatible exécution locale & Azure (DBFS / DataLake)
"""

import os
import sys
import csv
import psycopg2
import xml.etree.ElementTree as ET
from pathlib import Path
import re

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    pipeline,
)

# ─── NLTK punkt ───
for pkg, probe in [
    ("punkt",     "tokenizers/punkt/french.pickle"),
    ("punkt_tab", "tokenizers/punkt_tab/french")
]:
    try:
        nltk.data.find(probe)
    except LookupError:
        nltk.download(pkg)

# ──────────────────────────────────────────────────────────────
#                        Azure utilities
# ──────────────────────────────────────────────────────────────
class AzureUtils:
    def __init__(self, mount_dir="/mnt/data"):
        self.mount_dir = mount_dir

    def detect_azure_run(self) -> bool:
        args = dict(arg.split("=") for arg in sys.argv[1:] if "=" in arg)
        return args.get("AZURE_RUN", "false").lower() == "true"

    # (méthodes mount_dir_Azure et get_latest_xml inchangées …)

# ──────────────────────────────────────────────────────────────
#                      PostgreSQL utilities
# ──────────────────────────────────────────────────────────────
class PostgresUtils:
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

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host, database=self.database,
            user=self.user, password=self.password,
            port=self.port, sslmode="require"
        )
        print(f"[Postgres] Connexion OK → {self.database}")

    def create_table(self, table_name="textTimeline"):
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            docID INT,
            article TEXT,                 -- ← nouvelle colonne
            ordinal_prise INT,
            orateur TEXT,
            debut DOUBLE PRECISION,
            fin   DOUBLE PRECISION,
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

    def get_last_doc_id(self, table_name="textTimeline"):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT MAX(docID) FROM {table_name};")
            res = cur.fetchone()[0]
        return res if res is not None else 0

    # ————— Insertion filtrée / re-basée (article 9) —————
    def insert_csv(
        self,
        csv_path,
        table_name="textTimeline",
        target_article="9",
    ):
        new_doc_id   = self.get_last_doc_id(table_name) + 1
        rows_to_send = []

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("article") == target_article:
                    rows_to_send.append(row)

        if not rows_to_send:
            print(f"[Postgres] Aucun enregistrement pour l’article {target_article} – rien inséré.")
            return

        # re-base : première prise de l’article 9 → 0 s
        offset = float(rows_to_send[0]["debut"])

        with self.conn.cursor() as cur:
            for row in rows_to_send:
                debut = float(row["debut"]) - offset
                fin   = float(row["fin"])   - offset
                cur.execute(
                    f"""INSERT INTO {table_name}
                        (docID, article, ordinal_prise, orateur, debut, fin,
                         sad, disgust, angry, neutral, fear,
                         surprise, happy, texte)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);""",
                    (
                        new_doc_id, row["article"],
                        int(row["ordinal_prise"]), row["orateur"],
                        debut, fin,
                        float(row["sad"]),    float(row["disgust"]),
                        float(row["angry"]),  float(row["neutral"]),
                        float(row["fear"]),   float(row["surprise"]),
                        float(row["happy"]),  row["texte"]
                    )
                )
        self.conn.commit()
        print(f"[Postgres] Insertion terminée • docID={new_doc_id} • article={target_article}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("[Postgres] Connexion fermée.")

# ──────────────────────────────────────────────────────────────
#                   Text Emotion Analyzer class
# ──────────────────────────────────────────────────────────────
class TextEmotionAnalyzer:
    def __init__(self, model_dir: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Model] Loading CamemBERT from: {model_dir}")
        self.model     = CamembertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        self.pipe = pipeline("text-classification",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             top_k=5)

        self.rename_map = {
            "sad": "sad", "disgusted": "disgust", "anger": "angry",
            "neutral": "neutral", "fear": "fear",
            "surprise": "surprise", "joy": "happy",
        }
        self.ordered_labels = [
            "sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"
        ]

    # ───────── helpers ─────────
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

    # ───────── XML → CSV ─────────
    def analyze_xml(self, xml_path: str) -> Path:
        xml_path = Path(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"ns": "http://schemas.assemblee-nationale.fr/referentiel"}

        prises, current_article = [], None
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]   # suppr. namespace

            # mémorise l'article
            if tag == "point" and elem.attrib.get("art"):
                current_article = elem.attrib["art"].strip()

            # prise de parole
            if tag == "paragraphe" and "ordinal_prise" in elem.attrib:
                texte_elem = elem.find("ns:texte", ns)
                if texte_elem is None:
                    continue
                texte = "".join(texte_elem.itertext())
                texte = self._clean_text(texte)
                stime = texte_elem.attrib.get("stime")
                if not texte or stime is None:
                    continue
                prises.append({
                    "article": current_article or "",
                    "ordinal_prise": elem.attrib["ordinal_prise"],
                    "orateur": elem.find("ns:orateurs/ns:orateur/ns:nom", ns).text
                               if elem.find("ns:orateurs/ns:orateur/ns:nom", ns) is not None
                               else "Inconnu",
                    "texte": texte,
                    "debut": float(stime),
                })

        if not prises:
            raise ValueError("Aucune prise de parole extraite")

        prises = sorted(prises, key=lambda x: int(x["ordinal_prise"]))

        # calcul des fins (timeline d'origine)
        for i in range(len(prises) - 1):
            prises[i]["fin"] = prises[i + 1]["debut"]
        prises[-1]["fin"] = None
        prises = [p for p in prises if p["fin"] is not None]

        # — pipeline d'émotions —
        results = []
        for p in prises:
            agg = {}
            for chunk in self._chunk_sentences(p["texte"]):
                for pr in self.pipe(chunk)[0]:
                    agg[pr["label"]] = agg.get(pr["label"], 0) + pr["score"]
            total = sum(agg.values()) or 1
            agg = {k: v / total for k, v in agg.items()}
            mapped = {self.rename_map[lbl]: agg.get(lbl, 0.0) for lbl in self.rename_map}
            mapped["surprise"], mapped["neutral"] = mapped["neutral"], mapped["surprise"]
            results.append({**p, **mapped})

        df = pd.DataFrame(results)[
            ["article", "ordinal_prise", "orateur", "texte", "debut", "fin"] + self.ordered_labels
        ].fillna(0)

        out_path = self.output_dir / f"{xml_path.stem}_emotions.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[OK] CSV généré → {out_path}")
        return out_path

# ──────────────────────────────────────────────────────────────
#                            main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # --- chemins locaux d’exemple ; adaptez si besoin ---
    xml_local    = Path("input") / "CRSANR5L17S2025O1N250.xml"
    model_local  = Path("models")
    output_local = Path("output")
    output_local.mkdir(exist_ok=True)

    analyzer = TextEmotionAnalyzer(
        model_dir=str(model_local),
        output_dir=str(output_local),
    )
    csv_path = analyzer.analyze_xml(str(xml_local))

    pg = PostgresUtils()
    try:
        pg.connect()
        pg.create_table("textTimeline")
        pg.insert_csv(csv_path, "textTimeline", target_article="9")
    finally:
        pg.close()

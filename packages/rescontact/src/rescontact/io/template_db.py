# src/rescontact/templates/template_db.py
"""
Cache-first structure fetchers for PDB/AFDB.
Small, laptop-friendly: only downloads structures you actually need.

Cache layout (under RESCONTACT_TEMPLATE_DIR or .cache/rescontact/templates):
  templates/
    pdb/   <accession>.pdb or .cif
    afdb/  <uniprot>.pdb or .cif
"""

import os
from pathlib import Path
from typing import Optional

import requests

DEFAULT_CACHE = Path(os.environ.get("RESCONTACT_TEMPLATE_DIR", ".cache/rescontact/templates")).expanduser()
PDB_DIR = DEFAULT_CACHE / "pdb"
AFDB_DIR = DEFAULT_CACHE / "afdb"
for d in (PDB_DIR, AFDB_DIR):
    d.mkdir(parents=True, exist_ok=True)

def ensure_cache_dirs():
    return {"pdb": PDB_DIR, "afdb": AFDB_DIR}

def fetch_structure_cached(accession: str, source: str = "pdb", downloads_left_ref=None) -> Optional[Path]:
    """
    Try to find structure in cache; otherwise download.
    source: "pdb" or "afdb"
    """
    source = source.lower()
    if source not in ("pdb", "afdb"):
        return None

    if source == "pdb":
        # accession can be PDB id or Uniprot; try PDB id first
        pdb_id = accession.split("_")[0][:4].lower()
        for ext in (".pdb", ".cif", ".ent"):
            p = PDB_DIR / f"{pdb_id}{ext}"
            if p.exists():
                return p
        # try to download minimal .pdb
        if downloads_left_ref is not None and downloads_left_ref[0] <= 0:
            return None
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            r = requests.get(url, timeout=60)
            if r.ok and len(r.text) > 1000:
                p = PDB_DIR / f"{pdb_id}.pdb"
                p.write_text(r.text)
                return p
        except Exception:
            return None

    if source == "afdb":
        # assume accession like UniProt ID
        uniprot_id = accession.split("_")[0]
        for ext in (".pdb", ".cif"):
            p = AFDB_DIR / f"{uniprot_id}{ext}"
            if p.exists():
                return p
        if downloads_left_ref is not None and downloads_left_ref[0] <= 0:
            return None
        # try AFDB PDB
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            r = requests.get(url, timeout=60)
            if r.ok and len(r.text) > 1000:
                p = AFDB_DIR / f"{uniprot_id}.pdb"
                p.write_text(r.text)
                return p
        except Exception:
            return None

    return None

import json
from .cypher import SUBGRAPH_BY_PROTEIN
from ..db import Neo4jClient

def _to_dict(x):
    try:
        return dict(x)
    except Exception:
        return x

class SubgraphRetriever:
    def __init__(self, db: Neo4jClient):
        self.db = db

    def fetch_protein_context(self, uniprot_id: str) -> dict:
        rows = self.db.run(SUBGRAPH_BY_PROTEIN, {"uniprot_id": uniprot_id})
        if not rows:
            return {"uniprot_id": uniprot_id, "found": False}

        row = rows[0]
        return {
            "uniprot_id": uniprot_id,
            "found": True,
            "protein": _to_dict(row["p"]),
            "pockets": [_to_dict(x) for x in row["pockets"]],
            "residues": [_to_dict(x) for x in row["residues"]],
            "ligands": [_to_dict(x) for x in row["ligands"]],
            "bind_edges": [_to_dict(x) for x in row["binds"]],
            "tox": [_to_dict(x) for x in row["tox"]],
            "assays": [_to_dict(x) for x in row["assays"]],
            "papers": [_to_dict(x) for x in row["papers"]],
        }

    @staticmethod
    def as_text(context: dict) -> str:
        return json.dumps(context, indent=2, ensure_ascii=False)

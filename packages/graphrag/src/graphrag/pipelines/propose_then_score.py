import json
from datetime import datetime
from ..db import Neo4jClient
from ..llm import load_llm
from ..retrieval.subgraph_retriever import SubgraphRetriever

UPSERT_PRED_BIND = """
MATCH (pk:Pocket {pocket_id:$pocket_id})
MATCH (l:Ligand {ligand_id:$ligand_id})
MERGE (l)-[e:PREDICTED_BINDING {engine:$engine, created_at:$created_at}]->(pk)
SET e.score=$score, e.pose_path=$pose_path, e.constraints=$constraints
"""

def propose_then_score(
    db: Neo4jClient,
    llm_cfg,
    uniprot_id: str,
    pocket_id: str,
    docking_engine_name: str = "vina_stub"
):
    llm = load_llm(llm_cfg)
    retriever = SubgraphRetriever(db)

    ctx = retriever.fetch_protein_context(uniprot_id)
    ctx_text = retriever.as_text(ctx)

    prompt = (
        "You will be given a JSON evidence subgraph.\n"
        "Return JSON with keys candidates, questions_to_answer.\n\n"
        f"EVIDENCE_SUBGRAPH_JSON:\n{ctx_text}\n\n"
        "TASK: Propose candidate ligands/peptides with rationale and risks.\n"
    )
    proposal_raw = llm.generate(prompt)

    try:
        proposal = json.loads(proposal_raw)
    except Exception:
        proposal = {"candidates": [], "questions_to_answer": [], "raw": proposal_raw}

    # Stub scoring; replace with real docking call (vina/smina/gnina/diffdock)
    created_at = datetime.utcnow().isoformat()
    results = []
    for c in proposal.get("candidates", []):
        if c.get("type") != "ligand":
            continue
        ligand_id = c.get("id")
        if not ligand_id:
            continue

        score = -7.0  # placeholder

        db.run_write(UPSERT_PRED_BIND, {
            "pocket_id": pocket_id,
            "ligand_id": ligand_id,
            "engine": docking_engine_name,
            "created_at": created_at,
            "score": float(score),
            "pose_path": "",
            "constraints": json.dumps({"notes": "stub"}, ensure_ascii=False),
        })
        results.append({"ligand_id": ligand_id, "score": score})

    return {"proposal": proposal, "scored": results}

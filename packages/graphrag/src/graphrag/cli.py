import argparse
from .config import AppConfig
from .db import Neo4jClient
from .schema import apply_constraints
from .pipelines.propose_then_score import propose_then_score

def main():
    p = argparse.ArgumentParser("graphrag")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init-db")
    s.add_argument("--constraints", default="packages/graphrag/schema/constraints.cql")

    q = sub.add_parser("propose-then-score")
    q.add_argument("--uniprot-id", required=True)
    q.add_argument("--pocket-id", required=True)
    q.add_argument("--engine", default="vina_stub")

    args = p.parse_args()
    cfg = AppConfig()
    db = Neo4jClient(cfg.neo4j)
    try:
        if args.cmd == "init-db":
            apply_constraints(db, args.constraints)
            print("✅ Neo4j constraints/indexes applied.")
        elif args.cmd == "propose-then-score":
            out = propose_then_score(
                db=db,
                llm_cfg=cfg.llm,
                uniprot_id=args.uniprot_id,
                pocket_id=args.pocket_id,
                docking_engine_name=args.engine,
            )
            print(out)
    finally:
        db.close()

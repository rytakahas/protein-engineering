Extract entities from the provided text into JSON.

Entities:
- proteins (uniprot_id if present, gene, organism)
- ligands (name, smiles/inchikey if present)
- assays (metric/value/units)
- toxicity flags (type/severity)
- citations (doi/url)

Return JSON with keys: proteins, ligands, assays, toxicity, citations

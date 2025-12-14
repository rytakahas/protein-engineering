#!/usr/bin/env bash
set -euo pipefail
echo "Example loader stub. Provide CSVs and call ingest modules from your own driver."
echo "Suggested CSVs:"
echo " - targets.csv (uniprot_id,name,family,organism,gene,sequence,length)"
echo " - ligands.csv (ligand_id,name,smiles,inchi_key,scaffold_id,logp,tpsa,mw,source)"
echo " - assays.csv  (assay_id,uniprot_id,ligand_id,metric,value,units,system,conditions,source)"
echo " - toxicity.csv(tox_id,ligand_id,tox_type,severity,evidence,source)"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, gzip, io, sys, logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
from Bio.SeqUtils import seq1

log = logging.getLogger("pdb_to_fasta")

# Extendable map for modified residues → 1-letter
AA3_TO_1 = {
    # canonical
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K",
    "LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V",
    "TRP":"W","TYR":"Y",
    # common mods
    "MSE":"M","SEC":"U","PYL":"O","SEP":"S","TPO":"T","PTR":"Y","CSO":"C","HYP":"P","ASX":"B","GLX":"Z",
    # fallback handled by seq1(..., custom_map=...) which returns 'X' if unknown
}

def _open_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r", errors="ignore")

def parse_seqres_pdb(path: Path, min_len: int, max_len: Optional[int], verbose: bool=False) -> List[Tuple[str,str]]:
    """Extract sequences from SEQRES records in .pdb/.ent (fast, tolerant)."""
    if path.suffix.lower() not in (".pdb", ".ent"):
        return []
    chains: Dict[str, List[str]] = {}
    try:
        with _open_maybe_gzip(path) as fh:
            for line in fh:
                # PDB format: 'SEQRES' begins at col 1
                if not line.startswith("SEQRES"):
                    continue
                chain = (line[11].strip() or "X")
                # Residue names typically from col 20 onwards (space-separated 3-letter)
                toks = line[19:].split()
                aa = []
                for t in toks:
                    t = t.strip().upper()
                    if not t:
                        continue
                    try:
                        aa.append(seq1(t, custom_map=AA3_TO_1))
                    except Exception:
                        aa.append("X")
                chains.setdefault(chain, []).extend(aa)
    except Exception as e:
        if verbose:
            log.warning(f"[SEQRES] {path.name}: parse error: {e}")
        return []

    out = []
    for ch, letters in chains.items():
        seq = "".join(letters)
        L = len(seq)
        if L == 0 or L < min_len or (max_len and L > max_len):
            continue
        out.append((f"{path.stem}_{ch}", seq))
    if verbose:
        if out:
            log.info(f"[SEQRES] {path.name}: chains={len(out)} lengths={[len(s) for _,s in out]}")
        else:
            log.info(f"[SEQRES] {path.name}: no valid chains")
    return out

def parse_atoms_poly(path: Path, min_len: int, max_len: Optional[int], verbose: bool=False) -> List[Tuple[str,str]]:
    """Fallback: construct sequences from ATOM records (works for many coordinates)."""
    try:
        if path.suffix.lower() in (".pdb", ".ent", ".gz"):
            parser = PDBParser(PERMISSIVE=True, QUIET=True)
        else:
            parser = MMCIFParser(QUIET=True)
        with _open_maybe_gzip(path) as fh:
            structure = parser.get_structure(path.stem, fh)
    except Exception as e:
        if verbose:
            log.warning(f"[ATOM] {path.name}: parse error: {e}")
        return []

    ppb = PPBuilder()
    out: List[Tuple[str,str]] = []
    try:
        for model in structure:
            for chain in model:
                peptides = ppb.build_peptides(chain, aa_only=True)
                if not peptides:
                    continue
                seq = "".join(str(pp.get_sequence()) for pp in peptides)
                L = len(seq)
                if L == 0 or L < min_len or (max_len and L > max_len):
                    continue
                qid = f"{path.stem}_{getattr(chain, 'id', 'X')}"
                out.append((qid, seq))
    except Exception as e:
        if verbose:
            log.warning(f"[ATOM] {path.name}: PPBuilder error: {e}")
        return []

    if verbose:
        if out:
            log.info(f"[ATOM] {path.name}: chains={len(out)} lengths={[len(s) for _,s in out]}")
        else:
            log.info(f"[ATOM] {path.name}: no valid chains")
    return out

def collect_pdb_fastas(roots: List[Path], pdb_glob: str, recursive: bool,
                       limit_files: Optional[int], min_len: int, max_len: Optional[int],
                       verbose: bool=False) -> List[Tuple[str,str]]:
    # Find files (recursive if requested)
    patterns = ["*.pdb", "*.ent", "*.cif", "*.pdb.gz", "*.cif.gz"]
    files: List[Path] = []
    for r in roots:
        if not r.exists():
            if verbose:
                log.warning(f"[scan] root not found: {r}")
            continue
        for pat in patterns:
            if recursive:
                files.extend(sorted(r.rglob(pat)))
            else:
                files.extend(sorted(r.glob(pat)))

    # Optional limiting
    if limit_files is not None:
        files = files[:limit_files]

    if verbose:
        log.info(f"[scan] matched files: {len(files)}")

    seqs: List[Tuple[str,str]] = []
    for f in files:
        s = parse_seqres_pdb(f, min_len=min_len, max_len=max_len, verbose=verbose)
        if not s:
            s = parse_atoms_poly(f, min_len=min_len, max_len=max_len, verbose=verbose)
        seqs.extend(s)
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-root", nargs="+", required=True, help="One or more folders containing PDB/CIF files")
    ap.add_argument("--out", required=True, help="Output FASTA path")
    ap.add_argument("--pdb-glob", default="*", help="(unused now; kept for compatibility)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s", stream=sys.stdout
    )

    roots = [Path(p) for p in args.pdb_root]
    pairs = collect_pdb_fastas(
        roots, args.pdb_glob, args.recursive,
        args.limit_files, args.min_len, args.max_len, args.verbose
    )

    # De-duplicate identical sequences (keep first id)
    seen = set()
    out_lines = []
    for qid, seq in pairs:
        if seq in seen:
            continue
        seen.add(seq)
        out_lines.append(f">{qid}\n{seq}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(out_lines))

    print(f"[pdb_to_fasta] roots={len(roots)} files_limit={args.limit_files} recursive={args.recursive}")
    print(f"[pdb_to_fasta] sequences found={len(pairs)} unique_kept={len(seen)}")
    print(f"[pdb_to_fasta] wrote={args.out}")

if __name__ == "__main__":
    main()


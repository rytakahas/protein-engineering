# src/rescontact/api/monitor.py
import json, time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional
import numpy as np
from . import __package__ as _  # keep relative import happy
from rescontact.utils.psi import quantile_bins, hist_proportions, psi_from_props, categorize_psi

@dataclass
class MetricSpec:
    name: str
    extractor: Callable[[Dict[str, Any]], np.ndarray]  # ctx -> 1D array
    bins: int = 10
    use_fixed_edges: bool = True  # stick to baseline edges

@dataclass
class MetricState:
    edges: Optional[np.ndarray] = None
    base_props: Optional[np.ndarray] = None
    cur_counts: Optional[np.ndarray] = None
    n_cur: int = 0

class DriftMonitor:
    """
    Keeps per-metric baseline (edges+proportions) and live counts for current data.
    """
    def __init__(self, specs: List[MetricSpec]):
        self.specs = {s.name: s for s in specs}
        self.state: Dict[str, MetricState] = {s.name: MetricState() for s in specs}
        self.counters = {
            "requests_total": 0,
            "errors_total": 0,
            "cache_hits_esm": 0,
            "cache_hits_msa": 0,
            "degraded_fallbacks": 0,
        }
        self.latency_ms: List[float] = []

    # -------- Baseline I/O --------
    def fit_baseline(self, samples: Dict[str, np.ndarray]):
        for name, spec in self.specs.items():
            x = samples.get(name, np.array([]))
            edges = quantile_bins(x, spec.bins)
            base = hist_proportions(x, edges)
            self.state[name] = MetricState(edges=edges, base_props=base, cur_counts=np.zeros_like(base), n_cur=0)

    def save_baseline(self, path: str):
        blob = {}
        for name, st in self.state.items():
            if st.edges is None or st.base_props is None:
                continue
            blob[name] = {
                "edges": st.edges.tolist(),
                "base_props": st.base_props.tolist(),
            }
        with open(path, "w") as f:
            json.dump(blob, f)

    def load_baseline(self, path: str):
        with open(path, "r") as f:
            blob = json.load(f)
        for name, cfg in blob.items():
            edges = np.array(cfg["edges"], dtype=float)
            base = np.array(cfg["base_props"], dtype=float)
            self.state[name] = MetricState(edges=edges, base_props=base, cur_counts=np.zeros_like(base), n_cur=0)

    # -------- Live updates --------
    def observe(self, ctx: Dict[str, Any]):
        t0 = time.time()
        self.counters["requests_total"] += 1
        for name, spec in self.specs.items():
            st = self.state.get(name)
            if st is None or st.edges is None:
                continue
            x = spec.extractor(ctx)
            props = hist_proportions(x, st.edges)
            if st.cur_counts is None:
                st.cur_counts = np.zeros_like(props)
            st.cur_counts += props
            st.n_cur += 1
        self.latency_ms.append((time.time() - t0) * 1000.0)

    def snapshot(self) -> Dict[str, Any]:
        out = {"psi": {}, "counts": dict(self.counters), "latency_ms_p50_p95": None}
        if self.latency_ms:
            lat = np.array(self.latency_ms, dtype=float)
            out["latency_ms_p50_p95"] = [float(np.percentile(lat, 50)), float(np.percentile(lat, 95))]
        for name, st in self.state.items():
            if st.base_props is None or st.edges is None or st.cur_counts is None or st.n_cur == 0:
                continue
            q = st.cur_counts / max(st.n_cur, 1)  # average proportions
            p = st.base_props
            val = psi_from_props(p, q)
            out["psi"][name] = {
                "value": float(val),
                "category": categorize_psi(val),
                "bins": len(p),
            }
        return out

    def reset_current(self):
        for st in self.state.values():
            if st.cur_counts is not None:
                st.cur_counts[:] = 0
            st.n_cur = 0
        self.latency_ms.clear()

    # -------- Prometheus text exposition --------
    def prometheus(self) -> str:
        lines = []
        for k, v in self.counters.items():
            lines.append(f'# TYPE rescontact_{k} counter')
            lines.append(f"rescontact_{k} {int(v)}")
        snap = self.snapshot()
        if snap.get("latency_ms_p50_p95"):
            p50, p95 = snap["latency_ms_p50_p95"]
            lines.append("# TYPE rescontact_latency_ms gauge")
            lines.append(f"rescontact_latency_ms{{quantile=\"0.50\"}} {p50:.3f}")
            lines.append(f"rescontact_latency_ms{{quantile=\"0.95\"}} {p95:.3f}")
        for name, v in snap["psi"].items():
            lines.append("# TYPE rescontact_psi gauge")
            lines.append(f'rescontact_psi{{metric="{name}",category="{v["category"]}"}} {v["value"]:.6f}')
        return "\n".join(lines) + "\n"


# packages/resintnet/src/resintnet/memory_flow.py
import numpy as np
from numpy.linalg import solve

def build_laplacian(n, edges, C, Llen):
    """Unnormalized Laplacian for a weighted network."""
    L = np.zeros((n, n), dtype=float)
    for (i, j), c, l in zip(edges, C, Llen):
        w = c  # conductance is the weight in L
        L[i, i] += w; L[j, j] += w
        L[i, j] -= w; L[j, i] -= w
    return L

def solve_potential(L, q, outlet=0):
    """
    Solve L p = q with a Dirichlet gauge: fix one outlet node to p=0.
    Remove its row/col and solve reduced system; then reinsert p_outlet=0.
    """
    n = L.shape[0]
    keep = [k for k in range(n) if k != outlet]
    Lr = L[np.ix_(keep, keep)]
    qr = q[keep]
    pr = solve(Lr, qr)
    p = np.zeros(n); p[keep] = pr; p[outlet] = 0.0
    return p

def flows_from_potentials(edges, C, p):
    """Q_ij = C_ij * (p_i - p_j) for each link."""
    Q = np.zeros(len(edges), dtype=float)
    for idx, (i, j) in enumerate(edges):
        Q[idx] = C[idx] * (p[i] - p[j])
    return Q

def random_training_load(n, theta, strength=1.0, rng=None):
    """
    Simple directional load: put +strength on a band of sources and
    -strength on a sink band (opposite direction) determined by angle theta.
    For proteins you can map theta to e.g. a plane and split nodes.
    """
    rng = np.random.default_rng(None if rng is None else rng)
    # Example: embed nodes on a unit circle for demonstration
    # Replace with residue coordinates projected to 2D for real data.
    phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    direction = np.array([np.cos(theta), np.sin(theta)])
    xy = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    proj = xy @ direction
    q = np.zeros(n)
    q[proj > 0.2] = +strength / max(1, np.sum(proj > 0.2))
    q[proj < -0.2] = -strength / max(1, np.sum(proj < -0.2))
    # ensure sum(q)=0
    q -= q.mean()
    return q

def power_loss(Q, C):
    """E = sum(Q_ij^2 / C_ij)."""
    return np.sum((Q**2) / np.clip(C, 1e-12, None))

def update_conductances_discrete(edges, C, Llen, Q2_avg, gamma, K):
    """
    Eq. (4) discrete update:
    C*_{ij} ∝ <Q_ij^2>^{(γ+1)/2} / l_ij, then renormalize to satisfy
    K^γ = Σ (C_ij l_ij)^γ l_ij  (material constraint).
    """
    num = (Q2_avg ** ((gamma + 1.0) * 0.5)) / np.clip(Llen, 1e-12, None)
    # Normalize such that material constraint holds
    # Find scalar alpha s.t. K^γ = Σ (alpha * num * l)^γ * l
    l = Llen
    S = np.sum((num * l) ** gamma * l)
    if S <= 0:
        return C
    alpha = (K**gamma / S) ** (1.0 / gamma)
    C_new = alpha * num
    # Optional: clamp minimum/maximum conductance
    return np.clip(C_new, 1e-12, None)

def train_memory_network(n, edges, Llen, C0, gamma=0.5, K=1.0,
                         theta_train=0.0, T_avg=16, iters=200, outlet=0, rng=0):
    """
    Train with loads around theta_train, evolving C by Eq.(4).
    Returns final conductances and a function to measure power loss vs theta.
    """
    C = C0.copy()
    rng = np.random.default_rng(rng)

    for t in range(iters):
        # time-average Q^2 over T_avg random load fluctuations near theta_train
        Q2_sum = np.zeros(len(edges))
        for _ in range(T_avg):
            theta = theta_train + rng.normal(scale=0.05)  # small jitter
            q = random_training_load(n, theta, strength=1.0, rng=rng)
            L = build_laplacian(n, edges, C, Llen)
            p = solve_potential(L, q, outlet=outlet)
            Q = flows_from_potentials(edges, C, p)
            Q2_sum += Q**2
        Q2_avg = Q2_sum / T_avg
        C = update_conductances_discrete(edges, C, Llen, Q2_avg, gamma, K)

    def power_vs_theta(theta_list):
        vals = []
        for th in theta_list:
            q = random_training_load(n, th, strength=1.0, rng=rng)
            L = build_laplacian(n, edges, C, Llen)
            p = solve_potential(L, q, outlet=outlet)
            Q = flows_from_potentials(edges, C, p)
            vals.append(power_loss(Q, C))
        return np.array(vals)

    return C, power_vs_theta


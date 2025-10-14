from typing import Dict, List, Tuple
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None


EPS = 1e-8


def _to_matrix(vecs: List[np.ndarray]) -> np.ndarray:
    return np.vstack([v.astype(np.float32, copy=False) for v in vecs])


def compute_deltas(Y: np.ndarray) -> np.ndarray:
    return Y[1:] - Y[:-1]


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=1, keepdims=True) + EPS
    bn = np.linalg.norm(b, axis=1, keepdims=True) + EPS
    return np.sum(a * b, axis=1, keepdims=False) / (an[:, 0] * bn[:, 0])


def arclengths(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.array([0.0], dtype=np.float32)
    if len(X) == 1:
        return np.array([0.0], dtype=np.float32)
    d = np.linalg.norm(X[1:] - X[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s.astype(np.float32)


def resample_by_arclength(X: np.ndarray, M: int) -> np.ndarray:
    N = len(X)
    if M <= 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    if N == 0:
        return np.zeros((M, X.shape[1]), dtype=np.float32)
    if N == M:
        return X.copy()
    if N == 1:
        return np.repeat(X, M, axis=0)

    s = arclengths(X)
    total = float(s[-1])
    if total == 0.0:
        return np.repeat(X[:1], M, axis=0)

    targets = np.linspace(0.0, total, M, dtype=np.float32)
    Y = np.zeros((M, X.shape[1]), dtype=np.float32)

    j = 0
    for i, t in enumerate(targets):
        while j + 1 < N and s[j + 1] < t:
            j += 1
        if j + 1 == N:
            Y[i] = X[-1]
        else:
            s0, s1 = s[j], s[j + 1]
            w = 0.0 if s1 <= s0 else float((t - s0) / (s1 - s0))
            Y[i] = (1.0 - w) * X[j] + w * X[j + 1]
    return Y


def _align_sequences(
    AY: np.ndarray, AV: np.ndarray, *, order_V: int, align: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two sequences AY and AV based on the specified alignment strategy."""
    T = AY.shape[0]
    if AV.shape[0] == T:
        return AY, AV

    if align == "resample":
        PV = AV if order_V == 0 else integrate_k(AV, order_V)
        # Resample to T + order_V to get T points after k-diff
        PV_resampled = resample_by_arclength(PV, T + order_V)
        AV_aligned = PV_resampled if order_V == 0 else positions_to_kdiff(PV_resampled, order_V)
        # Ensure correct length after resampling and diffing
        min_len = min(AY.shape[0], AV_aligned.shape[0])
        return AY[:min_len], AV_aligned[:min_len]
    
    if align == "truncate":
        min_len = min(T, AV.shape[0])
        return AY[:min_len], AV[:min_len]

    raise ValueError(f"Length mismatch: A_Y={T}, A_V={AV.shape[0]} and align='{align}'")


def kdiff(Y: np.ndarray, k: int) -> np.ndarray:
    """Compute k-th forward difference along time axis for a sequence of positions Y."""
    if k < 0:
        raise ValueError("Order k must be non-negative.")
    if k == 0:
        return Y
    
    diff_Y = np.diff(Y, n=k, axis=0)
    
    if diff_Y.shape[0] == 0:
        return np.zeros((0, Y.shape[1]), dtype=Y.dtype)
        
    return diff_Y


def integrate_k(D: np.ndarray, k: int) -> np.ndarray:
    """Integrate k-th difference D back to pseudo-positions by k-fold cumulative sum.
    Returns a positions array whose length is len(D)+k.
    Starts from zeros for each integration.
    """
    if k <= 0:
        return D
    for _ in range(k):
        out = np.zeros((D.shape[0] + 1, D.shape[1]), dtype=np.float32)
        for t in range(D.shape[0]):
            out[t + 1] = out[t] + D[t]
        D = out
    return D


def positions_to_kdiff(P: np.ndarray, k: int) -> np.ndarray:
    """Apply k-fold differencing on positions P to get back k-th differences."""
    if k <= 0:
        return P
    for _ in range(k):
        P = compute_deltas(P)
    return P


def _normalize_Y_inputs_order(
    Ys: Dict[str, List[np.ndarray]] | List[List[np.ndarray]], k_order: int
) -> Tuple[List[str], List[np.ndarray]]:
    """Normalize inputs to (labels, list_of_ordered_sequences)."""
    if isinstance(Ys, dict):
        labels = list(Ys.keys())
        Y_lists = list(Ys.values())
    else:
        labels = [str(i) for i in range(len(Ys))]
        Y_lists = Ys

    seq_list = [_to_matrix(Y_list) for Y_list in Y_lists]
    if k_order > 0:
        seq_list = [kdiff(Y, k_order) for Y in seq_list]

    return labels, seq_list


def _align_sequence_pair(A: np.ndarray, B: np.ndarray, *, order: int, align: str = "truncate") -> Tuple[np.ndarray, np.ndarray]:
    """Align two sequences A, B of the same order (0=positions, k>0=k-th differences)."""
    if A.shape[0] == B.shape[0]:
        return A, B
    
    if align == "error":
        raise ValueError(f"Length mismatch: {A.shape[0]} vs {B.shape[0]}")

    if align == "resample":
        L = min(A.shape[0], B.shape[0])
        PA = A if order == 0 else integrate_k(A, order)
        PB = B if order == 0 else integrate_k(B, order)
        
        # Resample to L + order positions to get L points after k-diff
        PA_rs = resample_by_arclength(PA, L + order)
        PB_rs = resample_by_arclength(PB, L + order)
        
        A_al = PA_rs if order == 0 else positions_to_kdiff(PA_rs, order)
        B_al = PB_rs if order == 0 else positions_to_kdiff(PB_rs, order)
        
        # Truncate to ensure length consistency
        final_L = min(A_al.shape[0], B_al.shape[0])
        return A_al[:final_L], B_al[:final_L]

    # Default to truncate
    L = min(A.shape[0], B.shape[0])
    return A[:L], B[:L]


def pairwise_similarity(
    Ys: Dict[str, List[np.ndarray]] | List[List[np.ndarray]],
    *,
    order: int = 1,
    metric: str = "mean_cos",
    align: str = "truncate",
) -> Tuple[List[str], np.ndarray]:
    """
    General pairwise similarity across multiple answers at arbitrary difference order.
    - order=0: compare positions Y
    - order=1: compare first differences ΔY (default)
    - order>=2: compare k-th differences
    - metric: 'mean_cos' (mean cosine per aligned time step) or 'pearson' (flattened Pearson)
    - align: 'truncate' | 'resample' | 'error'
    Returns: (labels, similarity_matrix [N x N])
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    labels, seqs = _normalize_Y_inputs_order(Ys, order)
    N = len(labels)
    sim = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        sim[i, i] = 1.0
        for j in range(i + 1, N):
            Ai, Bj = _align_sequence_pair(seqs[i], seqs[j], order=order, align=align)
            if metric == "mean_cos":
                cos = cosine_sim_matrix(Ai, Bj)
                val = float(np.mean(cos)) if cos.size > 0 else 0.0
            elif metric == "pearson":
                a = Ai.reshape(-1)
                b = Bj.reshape(-1)
                if a.size == 0 or b.size == 0:
                    val = 0.0
                else:
                    a_mu = a - np.mean(a)
                    b_mu = b - np.mean(b)
                    num = float(np.dot(a_mu, b_mu))
                    den = float((np.linalg.norm(a_mu) + EPS) * (np.linalg.norm(b_mu) + EPS))
                    val = num / den
            else:
                raise ValueError("Unsupported metric. Use 'mean_cos' or 'pearson'.")
            sim[i, j] = sim[j, i] = val
    return labels, sim


def _normalize_Y_inputs(
    Ys: Dict[str, List[np.ndarray]] | List[List[np.ndarray]],
) -> Tuple[List[str], List[List[np.ndarray]]]:
    """Normalize inputs to (labels, list_of_Y_lists) without differencing."""
    if isinstance(Ys, dict):
        labels = list(Ys.keys())
        Y_lists = list(Ys.values())
    else:
        labels = [str(i) for i in range(len(Ys))]
        Y_lists = Ys
    return labels, Y_lists


def _menger_curvature_sequence(Y: np.ndarray) -> np.ndarray:
    """Compute Menger curvature per interior step for a sequence of positions Y.
    Returns array of shape [T-2], with κ_i for triplets (i-1, i, i+1).
    κ = 4A / (abc), where a=||p_i - p_{i-1}||, b=||p_{i+1} - p_i||, c=||p_{i+1} - p_{i-1}||
    and A is triangle area via Heron's formula.
    """
    T = Y.shape[0]
    if T < 3:
        return np.zeros((0,), dtype=np.float32)
    Pm1 = Y[:-2]
    P0 = Y[1:-1]
    P1 = Y[2:]
    a = np.linalg.norm(P0 - Pm1, axis=1)
    b = np.linalg.norm(P1 - P0, axis=1)
    c = np.linalg.norm(P1 - Pm1, axis=1)
    # Heron's formula for triangle area
    s = 0.5 * (a + b + c)
    # Numerical safety: clamp inside sqrt >= 0
    area_sq = np.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    A = np.sqrt(area_sq)
    denom = (a * b * c) + EPS
    kappa = 4.0 * A / denom
    # Replace any NaN with 0.0
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return kappa.astype(np.float32)


def pairwise_menger_curvature_similarity(
    Ys: Dict[str, List[np.ndarray]] | List[List[np.ndarray]],
    *,
    metric: str = "pearson",
    align: str = "truncate",
) -> Tuple[List[str], np.ndarray]:
    """
    Pairwise similarity using Menger curvature sequences.
    - Computes κ sequences per label (length T-2 per trajectory), aligning by truncation or resampling (resampling not supported; falls back to truncate).
    - metric:
        - 'pearson': Pearson correlation between curvature sequences (centered)
        - 'mean_cos': cosine similarity between raw curvature sequences
    Returns (labels, [N x N] similarity matrix).
    """
    labels, Ys_list = _normalize_Y_inputs(Ys)
    # Build curvature sequences
    C_list: List[np.ndarray] = []
    for Y_list in Ys_list:
        Y = _to_matrix(Y_list)
        C = _menger_curvature_sequence(Y)
        C_list.append(C)

    N = len(labels)
    S = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        S[i, i] = 1.0
        for j in range(i + 1, N):
            ci = C_list[i]
            cj = C_list[j]
            if align == "resample":
                # Not meaningful for 1D scalar signals here; fallback to truncate
                align_mode = "truncate"
            else:
                align_mode = "truncate"
            L = min(ci.shape[0], cj.shape[0])
            if L == 0:
                val = 0.0
            else:
                ai = ci[:L]
                bj = cj[:L]
                if metric == "pearson":
                    ai_mu = ai - float(np.mean(ai))
                    bj_mu = bj - float(np.mean(bj))
                    num = float(np.dot(ai_mu, bj_mu))
                    den = float((np.linalg.norm(ai_mu) + EPS) * (np.linalg.norm(bj_mu) + EPS))
                    val = num / den
                elif metric == "mean_cos":
                    num = float(np.dot(ai, bj))
                    den = float((np.linalg.norm(ai) + EPS) * (np.linalg.norm(bj) + EPS))
                    val = num / den
                else:
                    raise ValueError("Unsupported metric for curvature: use 'pearson' or 'mean_cos'")
            S[i, j] = S[j, i] = val
    return labels, S


def plot_similarity_heatmap(
    similarity: np.ndarray,
    labels: List[str] | None = None,
    title: str = "Similarity (mean cosine)",
    height: int = 800,
    width: int = 800,
    save_pdf_path: str | None = None,
    *,
    show_axis_text: bool = True,
    color_scale: str | list[str] = "RdBu_r",
):
    """
    Plot a heatmap of a similarity matrix using Plotly for better notebook UX.
    Optionally, download the heatmap as a PDF file.
    Falls back to printing a warning if Plotly is unavailable.
    """
    if px is None:
        print("plotly is not installed; cannot render heatmap.")
        return
    if labels is None:
        labels = [str(i) for i in range(similarity.shape[0])]
    fig = px.imshow(
        similarity,
        x=labels,
        y=labels,
        color_continuous_scale=color_scale,
        origin="lower",
        zmin=float(np.min(similarity)),
        zmax=float(np.max(similarity)),
        title=title,
        height=height,
        width=width,
        aspect="auto",
    )
    if not show_axis_text:
        try:
            fig.update_xaxes(showticklabels=False, ticks="", title_text="")
            fig.update_yaxes(showticklabels=False, ticks="", title_text="")
        except Exception:
            pass
    # fig.update_layout(height=height, width=width)
    # fig.show()
    if save_pdf_path is not None:
        try:
            # Requires kaleido or orca installed
            fig.write_image(save_pdf_path, format="pdf")
            print(f"PDF saved to {save_pdf_path}")
        except Exception as e:
            print(f"Could not save PDF: {e}")
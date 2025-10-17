import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_FEATURES = 50

# Étape 4 (heuristique) : PCA + KMeans pour shortlist de candidats
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =======================================================
# Chargement et validation des fichiers CSV
# =======================================================
def detect_sep(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
    return ';' if ';' in first else ','

def parse_vec_50(s: str) -> np.ndarray:
    s = str(s).strip()
    sep = ';' if ';' in s else ','
    arr = np.asarray([float(x) for x in s.split(sep) if x.strip() != ""], dtype=np.float64)
    if arr.size != NUM_FEATURES:
        raise ValueError(f"Vecteur invalide ({arr.size} != 50)")
    return arr

def load_points(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    feats = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    missing = ["node_id"] + [c for c in feats if c not in df.columns]
    if "node_id" not in df.columns or any(c not in df.columns for c in feats):
        raise KeyError(f"Colonnes manquantes dans points (attendues: node_id, feature_1..feature_50)")
    try:
        df[feats] = df[feats].astype(np.float64)
    except Exception:
        for c in feats:
            df[c] = df[c].astype(str).str.replace(',', '.').astype(np.float64)
    df["node_id"] = df["node_id"].astype(str)
    return df

def load_queries(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    for col in ("point_A", "Y_vector", "D"):
        if col not in df.columns:
            raise KeyError(f"Colonne manquante dans queries: '{col}'")
    return df


# =======================================================
# Calcul de distance pondérée (exacte)
# =======================================================
def dist2_vectorized(points: np.ndarray, A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """d2 = (P^2)·Y - 2·P·(Y⊙A) + (A^2)·Y (forme algébrique exacte, évite sqrt)"""
    YA = Y * A
    A2Y = float((A * A) @ Y)
    P2Y = (points * points) @ Y
    PAY = points @ YA
    return P2Y - 2.0 * PAY + A2Y


# =======================================================
# Recherche exacte optimisée (préfiltre Top-K)
# =======================================================
def search_fast(points: np.ndarray, node_ids, A: np.ndarray, Y: np.ndarray, D: float, topk: int = 12):
    D2 = D * D
    matches = []

    # Préfiltre exact sur K plus gros poids
    idx = np.argsort(-Y)[:topk]
    diff = points[:, idx] - A[idx]
    partial = np.sum((diff * diff) * Y[idx], axis=1)
    survivors = partial <= D2

    P = points[survivors]
    ids = [node_ids[i] for i, keep in enumerate(survivors) if keep]

    # Calcul complet exact sur les survivants
    d2 = dist2_vectorized(P, A, Y)
    mask = d2 <= D2
    idxs = np.nonzero(mask)[0]
    for i in idxs:
        matches.append((ids[i], float(np.sqrt(max(d2[i], 0.0)))))

    matches.sort(key=lambda x: (x[1], x[0]))
    return matches


# =======================================================
# Étape 4 — Index PCA + KMeans (shortlist candidats)
# =======================================================
def build_shortlist_index(points_df: pd.DataFrame, n_components: int = 10, seed: int = 42):
    """
    Construit un index pour shortlist :
      - PCA (50D -> rD) pour classement rapide
      - KMeans (k ≈ √N) pour segmenter l’espace
    Retour : dict {pca, centroids, members}
    """
    if not SKLEARN_OK:
        return None

    feats = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    X = points_df[feats].to_numpy(dtype=np.float64)

    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)

    N = Z.shape[0]
    k = max(4, int(np.sqrt(N)))
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = kmeans.fit_predict(Z)

    members = [np.where(labels == j)[0] for j in range(k)]
    centroids = kmeans.cluster_centers_
    return {"pca": pca, "centroids": centroids, "members": members}

def shortlist_indices_for_query(A_50d: np.ndarray, index: dict, M: int = 5) -> np.ndarray:
    """Projet A en PCA, prend les M clusters aux centroids les plus proches, retourne indices candidats."""
    pca = index["pca"]
    centroids = index["centroids"]
    members = index["members"]

    A_r = pca.transform(A_50d.reshape(1, -1))[0]
    diffs = centroids - A_r[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    M = min(M, len(d2))
    top = np.argpartition(d2, kth=M-1)[:M]

    if len(top) == 1:
        return members[top[0]]
    return np.unique(np.concatenate([members[j] for j in top], axis=0))


# =======================================================
# Visualisation du graphe (PCA + k-NN) + coloration des nœuds trouvés
# =======================================================
def _pairwise_dist2(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1)
    return norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)

def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]

def show_graph(points_df: pd.DataFrame, found_nodes=None, k: int = 8, max_nodes: int = 500, seed: int = 42):
    feats = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    X_full = points_df[feats].to_numpy(dtype=np.float64)
    ids_full = points_df["node_id"].astype(str).to_numpy()

    rng = np.random.default_rng(seed)
    if X_full.shape[0] > max_nodes:
        idx = rng.choice(X_full.shape[0], max_nodes, replace=False)
        X = X_full[idx]
        ids = ids_full[idx]
    else:
        X = X_full
        ids = ids_full

    D2 = _pairwise_dist2(X)
    np.fill_diagonal(D2, np.inf)
    k = max(1, min(k, X.shape[0]-1))
    nbrs = np.argpartition(D2, kth=k, axis=1)[:, :k]

    Z = _pca_2d(X)

    plt.figure(figsize=(8, 6))
    # Arêtes en arrière-plan
    for i in range(Z.shape[0]):
        xi, yi = Z[i]
        for j in nbrs[i]:
            xj, yj = Z[j]
            plt.plot([xi, xj], [yi, yj], linewidth=0.5, alpha=0.2, color='gray')

    # Nœuds "trouvés" en rouge si fournis
    if found_nodes is not None and len(found_nodes) > 0:
        found_mask = np.isin(ids, list(found_nodes))
        if np.any(found_mask):
            plt.scatter(Z[~found_mask, 0], Z[~found_mask, 1], s=25, alpha=0.85, edgecolors='k', linewidths=0.3)
            plt.scatter(Z[found_mask, 0], Z[found_mask, 1], s=60, c='red', edgecolors='k', linewidths=0.5, label='Nœuds trouvés')
        else:
            plt.scatter(Z[:, 0], Z[:, 1], s=25, alpha=0.85, edgecolors='k', linewidths=0.3)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=25, alpha=0.85, edgecolors='k', linewidths=0.3)

    plt.title(f"Graphe k-NN (k={k}) – PCA 2D ({Z.shape[0]} nœuds)")
    if found_nodes:
        plt.legend(loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =======================================================
# Orchestration principale
# =======================================================
def main():
    points_path = "adsSim_data_nodes (1).csv"
    queries_path = "queries_structured (1).csv"
    output_path = "responses.csv"

    print("📂 Lecture des fichiers CSV…")
    points_df = load_points(points_path)
    queries_df = load_queries(queries_path)
    print("✅ Fichiers chargés avec succès")

    # Données en mémoire
    feature_cols = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    points_mat_full = points_df[feature_cols].to_numpy(dtype=np.float64)
    node_ids_full = points_df["node_id"].astype(str).tolist()

    # Étape 4 : index PCA + KMeans pour shortlist (si scikit-learn dispo)
    shortlist_index = None
    if SKLEARN_OK:
        print("🧠 [Étape 4] Construction de l’index PCA+KMeans pour shortlist…")
        shortlist_index = build_shortlist_index(points_df, n_components=10, seed=42)
        print("✅ Index PCA+KMeans prêt")
    else:
        print("ℹ️ scikit-learn introuvable : la recherche se fera sans shortlist (pleine recherche exacte).")

    print("\n🧮 Lancement de la recherche exacte optimisée…")
    results = []
    for _, q in queries_df.iterrows():
        qid = str(q["point_A"])
        Y = parse_vec_50(q["Y_vector"])
        D = float(q["D"])

        if "A_vector" in q and not pd.isna(q["A_vector"]):
            A = parse_vec_50(q["A_vector"])
        else:
            # Génération déterministe si A_vector manquant
            h = hashlib.sha256(qid.encode('utf-8')).hexdigest()
            seed = int(h[:16], 16) % (2**32)
            rng = np.random.default_rng(seed)
            A = rng.uniform(0.0, 100.0, size=NUM_FEATURES)

        # --- Étape 4 : shortlist candidats par clusters proches (si index dispo) ---
        if shortlist_index is not None:
            cand_idx = shortlist_indices_for_query(A, shortlist_index, M=5)  # M réglable (5–8 conseillé)
            points_mat = points_mat_full[cand_idx]
            node_ids = [node_ids_full[i] for i in cand_idx]
        else:
            points_mat = points_mat_full
            node_ids = node_ids_full

        matches = search_fast(points_mat, node_ids, A, Y, D, topk=12)
        results.append((qid, D, matches))

    # Écriture du fichier de sortie
    rows = []
    for qid, D, matches in results:
        rows.append({
            "query_id": qid,
            "D": D,
            "num_matches": len(matches),
            "nodes": ";".join(n for n, _ in matches),
            "nodes_with_distance": ";".join(f"{n}:{d:.6f}" for n, d in matches),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"✅ Fichier généré : {output_path}")

    # Pop-up graphe avec coloration des nœuds trouvés
    print("\n📊 Affichage du graphe (fenêtre pop-up)…")
    found = set()
    for _, _, matches in results:
        for n, _ in matches:
            found.add(n)
    show_graph(points_df, found_nodes=found, k=8, max_nodes=500)
    print("🖼️ Graphe affiché avec succès !")


if __name__ == "__main__":
    main()
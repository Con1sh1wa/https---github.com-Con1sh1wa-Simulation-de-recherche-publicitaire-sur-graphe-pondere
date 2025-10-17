import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import sys

# Optionnel : accélération native avec Numba
try:
    from numba import njit, prange
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

NUM_FEATURES = 50


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
    df[feats] = df[feats].astype(float)
    df["node_id"] = df["node_id"].astype(str)
    return df

def load_queries(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    return df


# =======================================================
# Calcul de distance pondérée (exacte)
# =======================================================
def dist2_vectorized(points, A, Y):
    YA = Y * A
    A2Y = float((A * A) @ Y)
    P2Y = (points * points) @ Y
    PAY = points @ YA
    return P2Y - 2.0 * PAY + A2Y


# =======================================================
# Recherche exacte optimisée
# =======================================================
def search_fast(points, node_ids, A, Y, D, topk=12):
    D2 = D * D
    matches = []

    # Préfiltre exact : on regarde d’abord les K dimensions les plus lourdes
    idx = np.argsort(-Y)[:topk]
    diff = points[:, idx] - A[idx]
    partial = np.sum((diff * diff) * Y[idx], axis=1)
    survivors = partial <= D2

    points = points[survivors]
    ids = [node_ids[i] for i, keep in enumerate(survivors) if keep]

    # Calcul complet exact sur les survivants
    d2 = dist2_vectorized(points, A, Y)
    mask = d2 <= D2
    idxs = np.nonzero(mask)[0]

    for i in idxs:
        matches.append((ids[i], float(np.sqrt(max(d2[i], 0.0)))))

    matches.sort(key=lambda x: (x[1], x[0]))
    return matches


# =======================================================
# Visualisation du graphe (PCA + k-NN)
# =======================================================
def show_graph(points_df, found_nodes=None, k=8, max_nodes=500):
    feats = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    X = points_df[feats].to_numpy(dtype=np.float64)
    ids = points_df["node_id"].astype(str).to_numpy()

    # Échantillonner pour lisibilité
    if X.shape[0] > max_nodes:
        idx = np.random.choice(X.shape[0], max_nodes, replace=False)
        X = X[idx]
        ids = ids[idx]

    # Projection PCA 2D via SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]

    # Distances euclidiennes pour les arêtes
    norms = np.sum(X * X, axis=1)
    D2 = norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, np.inf)
    nbrs = np.argpartition(D2, kth=k, axis=1)[:, :k]

    plt.figure(figsize=(8, 6))
    for i in range(Z.shape[0]):
        xi, yi = Z[i]
        for j in nbrs[i]:
            xj, yj = Z[j]
            plt.plot([xi, xj], [yi, yj], 'k-', alpha=0.2, linewidth=0.4)
    # Si des nœuds trouvés sont fournis, on les colore en rouge
    if found_nodes is not None:
        found_mask = np.isin(ids, list(found_nodes))
    plt.scatter(Z[found_mask, 0], Z[found_mask, 1],
                s=60, c='red', edgecolors='k', label='Nœuds trouvés')

    plt.scatter(Z[:, 0], Z[:, 1], s=30, c='cornflowerblue', edgecolors='k')
    plt.title(f"Graphe k-NN (k={k}) – PCA 2D ({len(Z)} nœuds)")
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

    print(f"📂 Lecture des fichiers CSV...")
    points_df = load_points(points_path)
    queries_df = load_queries(queries_path)
    print("✅ Fichiers chargés avec succès")

    print("\n🧮 Lancement de la recherche exacte optimisée...")
    feature_cols = [f"feature_{i+1}" for i in range(NUM_FEATURES)]
    points_mat = points_df[feature_cols].to_numpy(dtype=np.float64)
    node_ids = points_df["node_id"].astype(str).tolist()

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

        matches = search_fast(points_mat, node_ids, A, Y, D)
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

    print("\n📊 Affichage du graphe (fenêtre pop-up)...")
    # On colore les nœuds trouvés (tous ceux présents dans les résultats)
    found = set()
    for _, _, matches in results:
        for n, _ in matches:
            found.add(n)
    show_graph(points_df, found_nodes=found)

print("🖼️ Graphe affiché avec succès !")


if __name__ == "__main__":
    main()

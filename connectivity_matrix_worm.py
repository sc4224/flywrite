import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def build_from_fixed_layout(excel_path: str, sheet_name: str):
    """
    Fixed layout assumptions (1-indexed Excel coordinates):
    - Row 3 contains column neuron names (starting at Col D).
    - Col C contains row neuron names (starting at Row 4).
    - Data block starts at (Row 4, Col D).

    In 0-indexed pandas iloc:
    - header row index = 2
    - row-label col index = 2  (Col C)
    - data start col index = 3 (Col D)
    - data start row index = 3 (Row 4)
    """
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=object)

    header_r = 2
    rowlabel_c = 2
    data_r0 = 3
    data_c0 = 3

    # Column neuron names (postsynaptic): row 3, from col D onward
    col_labels = raw.iloc[header_r, data_c0:].tolist()
    col_labels = [str(x).strip() for x in col_labels]

    # Drop trailing empties in column labels, and slice data accordingly
    valid_col_mask = [(x != "" and x.lower() != "nan") for x in col_labels]
    if any(valid_col_mask):
        last_valid = max(i for i, ok in enumerate(valid_col_mask) if ok)
        col_labels = col_labels[: last_valid + 1]
    else:
        raise RuntimeError("No valid column neuron names found in Row 3 (from Col D onward).")

    # Row neuron names (presynaptic): col C, from row 4 downward
    row_labels = raw.iloc[data_r0:, rowlabel_c].tolist()
    row_labels = [str(x).strip() for x in row_labels]

    # Drop trailing empties in row labels, and slice data accordingly
    valid_row_mask = [(x != "" and x.lower() != "nan") for x in row_labels]
    if any(valid_row_mask):
        last_valid_r = max(i for i, ok in enumerate(valid_row_mask) if ok)
        row_labels = row_labels[: last_valid_r + 1]
    else:
        raise RuntimeError("No valid row neuron names found in Col C (from Row 4 downward).")

    # Data block: from row 4, col D; match trimmed label lengths
    data = raw.iloc[data_r0 : data_r0 + len(row_labels), data_c0 : data_c0 + len(col_labels)]
    data = data.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Build rectangular df (pre rows × post cols)
    df_rect = pd.DataFrame(data.to_numpy(), index=row_labels, columns=col_labels)

    # Convert to binary presence
    df_bin = (df_rect > 0).astype(np.int8)

    # Make square on the union of labels (so it matches FlyWire-style square adjacency)
    nodes = list(dict.fromkeys(list(df_bin.columns) + list(df_bin.index)))  # preserve order, unique
    name_to_idx = {name: i for i, name in enumerate(nodes)}

    # Reindex to full square (missing rows/cols -> 0)
    square = df_bin.reindex(index=nodes, columns=nodes, fill_value=0).to_numpy(dtype=np.int8)

    # Build sparse csr
    row_idx, col_idx = np.where(square > 0)
    A = csr_matrix((np.ones(len(row_idx), dtype=np.int8), (row_idx, col_idx)),
                   shape=(len(nodes), len(nodes)), dtype=np.int8)
    if A.nnz:
        A.data[:] = 1

    return A, nodes, name_to_idx


def save_outputs(A: csr_matrix, nodes: list[str], name_to_idx: dict[str, int], out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / "worm_herm_chemical_binary_connectivity.npz"
    json_path = out / "worm_neuron_to_index_mapping.json"
    edges_path = out / "worm_herm_chemical_edges_binary.csv"

    save_npz(npz_path, A)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(name_to_idx, f, ensure_ascii=False, indent=2)

    coo = A.tocoo()
    pre = [nodes[i] for i in coo.row]
    post = [nodes[j] for j in coo.col]
    edges_df = pd.DataFrame({"pre": pre, "post": post, "value": np.ones(len(pre), dtype=np.int8)})
    edges_df.to_csv(edges_path, index=False)

    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {edges_path}")
    print(f"Matrix shape: {A.shape}, nnz(edges): {A.nnz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to 'SI 5 Connectome adjacency matrices.xlsx'")
    ap.add_argument("--sheet", default="hermaphrodite chemical", help="Sheet name to use")
    ap.add_argument("--out_dir", default="worm_flywire_format_out", help="Output directory")
    args = ap.parse_args()

    A, nodes, name_to_idx = build_from_fixed_layout(args.excel, args.sheet)
    save_outputs(A, nodes, name_to_idx, args.out_dir)


if __name__ == "__main__":
    main()
"""
build_signed_edges.py  –  create a CSV with (metabolite_i, metabolite_j, sign)

• sign = –1  for substrate→product pairs
• sign = +1  for pairs on the same side of the reaction
"""
import re, sys, requests, pathlib, csv
from collections import defaultdict

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def fetch_kegg_entry(kegg_id: str) -> str:
    """Try KEGG REST first; fall back to local flat file."""
    try:                                        # online path
        url = f"https://rest.kegg.jp/get/{kegg_id}"
        return requests.get(url, timeout=10).text
    except Exception:
        local = pathlib.Path("kegg_flat") / f"{kegg_id}.txt"
        if local.exists():
            return local.read_text()
        raise RuntimeError(f"cannot fetch {kegg_id}")

def parse_equation(txt: str):
    """Return (substrates[], products[], reversibleFlag)."""
    eq_line = next((l for l in txt.splitlines() if l.startswith("EQUATION")), "")
    if not eq_line:
        return [], [], True
    eq = eq_line.replace("EQUATION", "").strip()
    rev = "<=>" in eq
    lhs, rhs = re.split("<=>|=>", eq)

    def split(side):
        out = []
        for token in side.split("+"):
            token = token.strip()
            token = re.sub(r"^\d+\s+", "", token)   # drop stoich coeff.
            out.append(token)
        return out

    return split(lhs), split(rhs), rev

def build_signed_edges(substr, prod, rev):
    """Produce (u,v,sign) tuples."""
    edges = []
    # –1 edges across the arrow
    for s in substr:
        for p in prod:
            edges.append((s, p, -1))
            if rev:
                edges.append((p, s, -1))
    # +1 edges on the same side
    for group in (substr, prod):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.extend([(group[i], group[j], +1),
                              (group[j], group[i], +1)])
    return edges

# ------------------------------------------------------------
# main logic
# ------------------------------------------------------------
def process_reactions(reaction_ids: list[str]) -> dict[tuple[str, str], int]:
    """
    Processes a list of KEGG reaction IDs to build a dictionary of signed edges.

    Args:
        reaction_ids: A list of KEGG reaction IDs (e.g., "R00200").

    Returns:
        A dictionary where keys are (metabolite_i, metabolite_j) tuples
        and values are the sign (-1 or +1).
    """
    edge_sign = {}                         # {(u,v): sign}

    for rid in reaction_ids:
        try:
            txt = fetch_kegg_entry(rid)
            subs, prods, rev = parse_equation(txt)
            for u, v, s in build_signed_edges(subs, prods, rev):
                edge_sign[(u, v)] = s      # keep latest (overwrites duplicates)
        except Exception as e:
            print(f"Warning: Could not process {rid}. Error: {e}", file=sys.stderr)

    return edge_sign

def write_signed_edges_csv(edge_sign: dict[tuple[str, str], int], output_path: str | pathlib.Path):
    """Writes the signed edge data to a CSV file."""
    outfile = pathlib.Path(output_path)
    with outfile.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metabolite_i", "metabolite_j", "sign"])
        for (u, v), s in edge_sign.items():
            w.writerow([u, v, s])
    print(f"✅  wrote {len(edge_sign)} edges to {outfile.resolve()}")


# Example usage (can be removed or kept behind __name__ == "__main__" if needed for testing)
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         sys.exit(f"Usage: python {sys.argv[0]} R00200 R01786 ... <output_file.csv>")
#
#     rids_to_process = sys.argv[1:-1]
#     output_filename = sys.argv[-1]
#
#     if not rids_to_process or not output_filename.endswith(".csv"):
#         sys.exit(f"Usage: python {sys.argv[0]} R00200 R01786 ... <output_file.csv>")
#
#     print(f"Processing reactions: {rids_to_process}")
#     edges = process_reactions(rids_to_process)
#     write_signed_edges_csv(edges, output_filename)
#     print("Processing complete.")

# ------------------------------------------------------------
# old script execution block (removed)
# ------------------------------------------------------------
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         sys.exit("usage: build_signed_edges.py  R00200 R01786 ...")
#
#     edge_sign = {}                         # {(u,v): sign}
#
#     for rid in sys.argv[1:]:
#         txt = fetch_kegg_entry(rid)
#         subs, prods, rev = parse_equation(txt)
#         for u, v, s in build_signed_edges(subs, prods, rev):
#             edge_sign[(u, v)] = s          # keep latest (overwrites duplicates)
#
#     outfile = pathlib.Path("kegg_signed_edges.csv")
#     with outfile.open("w", newline="") as fh:
#         w = csv.writer(fh)
#         w.writerow(["metabolite_i", "metabolite_j", "sign"])
#         for (u, v), s in edge_sign.items():
#             w.writerow([u, v, s])
#
#     print(f"✅  wrote {len(edge_sign)} edges to {outfile.resolve()}")
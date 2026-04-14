import sqlite3
import csv
import math
from Bio import Align
from collections import defaultdict

def get_sequences(cursor):
    print("Fetching sequences...")
    cursor.execute("""
        SELECT td.chembl_id, cs.sequence
        FROM component_sequences cs
        JOIN target_components tc ON cs.component_id = tc.component_id
        JOIN target_dictionary td ON tc.tid = td.tid
        WHERE td.target_type = 'SINGLE PROTEIN'
    """)
    return {row[0]: row[1] for row in cursor.fetchall() if row[1]}

def get_similarity(seq1, seq2, aligner):
    if not seq1 or not seq2:
        return 0.0
    if seq1 == seq2:
        return 1.0
    try:
        score = aligner.score(seq1, seq2)
        max_score = min(len(seq1), len(seq2)) * aligner.match_score
        return score / max_score
    except Exception:
        return 0.0

def calculate_r2(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(i**2 for i in x)
    sum_y2 = sum(i**2 for i in y)
    sum_xy = sum(i*j for i, j in zip(x, y))
    numerator = n * sum_xy - sum_x * sum_y
    term1 = n * sum_x2 - sum_x**2
    term2 = n * sum_y2 - sum_y**2
    denominator = math.sqrt(max(0, term1) * max(0, term2))
    if denominator == 0:
        return 0.0
    r = numerator / denominator
    return r**2

def get_compound_data(target_info, molregnos):
    grouped = defaultdict(set)
    for c in target_info:
        if c['molregno'] in molregnos:
            grouped[c['molregno']].add(c['assay_id'])
    return grouped

def main():
    db_path = 'chembl_36.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    sequences = get_sequences(cursor)
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.1

    print("Fetching activity data...")
    query = """
    SELECT 
        d.chembl_id, d.doi, cs.canonical_smiles, a.molregno, td.chembl_id, td.pref_name, a.pchembl_value, s.chembl_id
    FROM activities a
    JOIN assays s ON a.assay_id = s.assay_id
    JOIN docs d ON s.doc_id = d.doc_id
    JOIN compound_structures cs ON a.molregno = cs.molregno
    JOIN target_dictionary td ON s.tid = td.tid
    WHERE a.pchembl_value IS NOT NULL
      AND td.target_type = 'SINGLE PROTEIN'
      AND d.doi IS NOT NULL
    """
    cursor.execute(query)
    
    data = defaultdict(lambda: defaultdict(list))
    target_names = {}
    print("Organizing data by document and target...")
    for row in cursor.fetchall():
        doc_id, doi, smiles, molregno, target_id, target_name, pchembl, assay_id = row
        target_names[target_id] = target_name
        data[doc_id][target_id].append({
            'molregno': molregno,
            'pchembl': pchembl,
            'smiles': smiles,
            'doi': doi,
            'assay_id': assay_id
        })
    
    results = []
    similarity_cache = {}
    set_counter = 0
    
    print(f"Processing {len(data)} documents...")
    doc_count = 0
    for doc_id, targets in data.items():
        doc_count += 1
        if doc_count % 1000 == 0:
            print(f"Processed {doc_count} documents...")
            
        target_ids = list(targets.keys())
        if len(target_ids) < 2:
            continue
            
        for i in range(len(target_ids)):
            for j in range(i + 1, len(target_ids)):
                tid1 = target_ids[i]
                tid2 = target_ids[j]
                
                compounds1 = {c['molregno']: c['pchembl'] for c in targets[tid1]}
                compounds2 = {c['molregno']: c['pchembl'] for c in targets[tid2]}
                shared_molregnos = set(compounds1.keys()) & set(compounds2.keys())
                
                if len(shared_molregnos) < 10:
                    continue

                pair_key = tuple(sorted((tid1, tid2)))
                if pair_key in similarity_cache:
                    sim = similarity_cache[pair_key]
                else:
                    seq1 = sequences.get(tid1)
                    seq2 = sequences.get(tid2)
                    if not seq1 or not seq2:
                        sim = 0.0
                    elif tid1 == tid2 or seq1 == seq2:
                        sim = 1.0
                    else:
                        sim = get_similarity(seq1, seq2, aligner)
                    similarity_cache[pair_key] = sim
                
                if sim < 0.9:
                    x = [compounds1[m] for m in shared_molregnos]
                    y = [compounds2[m] for m in shared_molregnos]
                    r2 = calculate_r2(x, y)
                    
                    if r2 >= 0.6:
                        set_counter += 1
                        assays1 = get_compound_data(targets[tid1], shared_molregnos)
                        assays2 = get_compound_data(targets[tid2], shared_molregnos)
                        smiles_map = {c['molregno']: c['smiles'] for c in targets[tid1] if c['molregno'] in shared_molregnos}
                        doi = targets[tid1][0]['doi']
                        
                        for m in shared_molregnos:
                            results.append([set_counter, smiles_map[m], m, tid1, target_names[tid1], compounds1[m], sim, r2, doc_id, doi, "|".join(sorted(list(assays1[m])))])
                            results.append([set_counter, smiles_map[m], m, tid2, target_names[tid2], compounds2[m], sim, r2, doc_id, doi, "|".join(sorted(list(assays2[m])))])

    print(f"Writing {len(results)} rows to results.csv...")
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set_id', 'canonical_smiles', 'molregno', 'target_id', 'target_name', 'pchembl', 'seq_similarity', 'r2', 'doc_id', 'doi', 'assay_ids'])
        writer.writerows(results)

if __name__ == "__main__":
    main()

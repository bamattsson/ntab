from typing import Optional, Union

import psycopg2


class ChEMBLRequester:
    """Client for querying a local ChEMBL PostgreSQL database."""

    def __init__(self, host: str, user: str, password: str, dbname: str) -> None:
        self.conn = psycopg2.connect(
            dbname=dbname,
            host=host,
            user=user,
            password=password,
        )
        self.cur = self.conn.cursor()
        self.dbname = dbname

    def get_chembl_id_to_smiles(self) -> list[dict[str, str | int | None]]:
        """Return all molecules with SMILES, earliest publication year, MW, and molecule type.

        cpd_earliest_year is the minimum publication year across all compound records associated
        with the molecule, or None if no year is available.

        mw_freebase is the molecular weight of the parent compound (salt/counterion stripped),
        from compound_properties. None if not available.

        molecule_type is the ChEMBL molecule type (e.g. 'Small molecule', 'Protein'), from
        molecule_dictionary. None if not set.
        """
        query = """
        SELECT md.chembl_id, cs.canonical_smiles, MIN(d.year) AS cpd_earliest_year,
               cp.mw_freebase, md.molecule_type
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN compound_records cr ON cr.molregno = md.molregno
        LEFT JOIN docs d ON d.doc_id = cr.doc_id
        LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
        GROUP BY md.chembl_id, cs.canonical_smiles, cp.mw_freebase, md.molecule_type
        """
        col_order = ["chembl_id", "canonical_smiles", "cpd_earliest_year", "mw_freebase", "molecule_type"]
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return [{k: v for k, v in zip(col_order, row)} for row in rows]
    
    def _get_component_classification(self) -> dict[int, tuple[str | None, str | None]]:
        """Return {component_id: (target_class, target_family)} for all components.

        Walks the protein_classification hierarchy from leaf to root using chained LEFT JOINs
        (up to 7 levels, matching ChEMBL's hierarchy depth). After reversing the path and
        removing the root node ('Protein Class'), target_class is the first remaining level
        and target_family the second. For components with multiple classification paths,
        the first path returned by the DB is used.
        """
        query = """
        SELECT
            cc.component_id,
            pc1.pref_name,
            pc2.pref_name,
            pc3.pref_name,
            pc4.pref_name,
            pc5.pref_name,
            pc6.pref_name,
            pc7.pref_name
        FROM component_class cc
        JOIN protein_classification pc1 ON pc1.protein_class_id = cc.protein_class_id
        LEFT JOIN protein_classification pc2 ON pc2.protein_class_id = pc1.parent_id
        LEFT JOIN protein_classification pc3 ON pc3.protein_class_id = pc2.parent_id
        LEFT JOIN protein_classification pc4 ON pc4.protein_class_id = pc3.parent_id
        LEFT JOIN protein_classification pc5 ON pc5.protein_class_id = pc4.parent_id
        LEFT JOIN protein_classification pc6 ON pc6.protein_class_id = pc5.parent_id
        LEFT JOIN protein_classification pc7 ON pc7.protein_class_id = pc6.parent_id
        """
        self.cur.execute(query)
        result: dict[int, tuple[str | None, str | None]] = {}
        for row in self.cur.fetchall():
            component_id = row[0]
            if component_id in result:
                continue  # keep first classification path per component
            leaf_to_root = [name for name in row[1:] if name is not None]
            root_to_leaf = list(reversed(leaf_to_root))[1:]  # drop root "Protein Class"
            target_class = root_to_leaf[0] if len(root_to_leaf) > 0 else None
            target_family = root_to_leaf[1] if len(root_to_leaf) > 1 else None
            result[component_id] = (target_class, target_family)
        return result

    def get_single_protein_targets(self) -> list[dict[str, str | None]]:
        """Return all single-protein targets with sequence, gene name, and protein classification.

        target_class and target_family are the two levels just below the root 'Protein Class'
        node in ChEMBL's protein classification hierarchy.

        Returns:
            List of dicts with keys: target_chembl_id, target_name, organism, uniprot_id,
            sequence, gene_name, target_class, target_family.
        """
        query = """
        SELECT
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.organism,
            cs.accession AS uniprot_id,
            cs.component_id,
            (
                SELECT csyn.component_synonym
                FROM component_synonyms csyn
                WHERE csyn.component_id = cs.component_id AND csyn.syn_type = 'GENE_SYMBOL'
                LIMIT 1
            ) AS gene_name,
            cs.sequence
        FROM target_dictionary td
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences cs ON tc.component_id = cs.component_id
        WHERE td.target_type = 'SINGLE PROTEIN'
            AND cs.component_type = 'PROTEIN'
        """
        col_order = ["target_chembl_id", "target_name", "organism", "uniprot_id", "component_id", "gene_name", "sequence"]
        self.cur.execute(query)
        rows = self.cur.fetchall()

        final_col_order = [
            "target_chembl_id", "uniprot_id", "gene_name",
            "target_class", "target_family", "organism", "target_name", "sequence",
        ]
        classification = self._get_component_classification()
        results = []
        for row in rows:
            record = {k: v for k, v in zip(col_order, row)}
            component_id = record.pop("component_id")
            target_class, target_family = classification.get(component_id, (None, None))
            record["target_class"] = target_class
            record["target_family"] = target_family
            results.append({k: record[k] for k in final_col_order})
        return results

    def get_all_single_protein_activity_data(
            self,
            target_chembl_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Union[str, float, None]]]:
        """Return activity data for single-protein binding assays at maximum confidence.

        Filters applied:
        - target_type = 'SINGLE PROTEIN'
        - assay_type = 'B' (binding)
        - confidence_score = 9 (maximum)
        - standard_type IN ('Ki', 'Kd', 'IC50')
        - data_validity_comment IS NULL or 'Manually validated' (excludes unreliable entries)
        - relationship_type IN ('D', 'H') (direct or homologue mappings only)

        Mutation annotations from variant_sequences are included when available (LEFT JOIN).

        Args:
            target_chembl_ids: Optional list of ChEMBL target IDs to restrict the query.

        Returns:
            List of dicts, one per activity measurement.
        """
        query = """
        SELECT
            td.chembl_id AS target_chembl_id,
            a2.chembl_id AS assay_chembl_id,
            md.chembl_id AS ligand_chembl_id,
            a.standard_type,
            a.standard_relation,
            a.pchembl_value,
            a.standard_value,
            a.standard_units,
            d.year AS doc_year,
            a.data_validity_comment,
            a.potential_duplicate,
            vs.mutation
        FROM activities a
        JOIN assays a2 ON a2.assay_id = a.assay_id
        JOIN molecule_dictionary md ON md.molregno = a.molregno
        JOIN target_dictionary td ON a2.tid = td.tid
        JOIN docs d ON a2.doc_id = d.doc_id
        LEFT JOIN variant_sequences vs ON a2.variant_id = vs.variant_id
        WHERE td.target_type = 'SINGLE PROTEIN'
            AND a2.assay_type = 'B'
            AND a2.confidence_score = 9
            AND a2.relationship_type IN ('D', 'H')
            AND a.standard_type IN ('Ki', 'Kd', 'IC50')
            AND (a.data_validity_comment IS NULL OR a.data_validity_comment = 'Manually validated')
        """
        col_order = [
            "target_chembl_id", "assay_chembl_id", "ligand_chembl_id",
            "standard_type", "standard_relation", "pchembl_value",
            "standard_value", "standard_units", "doc_year",
            "data_validity_comment", "potential_duplicate", "mutation",
        ]
        params = None
        if target_chembl_ids is not None:
            query += " AND td.chembl_id IN %s"
            params = (tuple(target_chembl_ids),)

        self.cur.execute(query, params)
        rows = self.cur.fetchall()
        return [{k: v for k, v in zip(col_order, row)} for row in rows]

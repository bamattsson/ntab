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
        """Return all molecules with SMILES and earliest publication year.

        earliest_year is the minimum publication year across all compound records associated
        with the molecule, or None if no year is available.
        """
        query = """
        SELECT md.chembl_id, cs.canonical_smiles, MIN(d.year) AS earliest_year
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN compound_records cr ON cr.molregno = md.molregno
        LEFT JOIN docs d ON d.doc_id = cr.doc_id
        GROUP BY md.chembl_id, cs.canonical_smiles
        """
        col_order = ["chembl_id", "canonical_smiles", "earliest_year"]
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return [{k: v for k, v in zip(col_order, row)} for row in rows]
    
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

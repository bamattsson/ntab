# timesplit-affinity-benchmark

Code for generating a time-split, novelty-filtered benchmark for protein-ligand binding affinity prediction.

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

```bash
uv pip install -e .
```

Or into a fresh virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Reproducing the data

### Set up ChEMBL

The benchmark is generated from a local ChEMBL PostgreSQL database. We used ChEMBL 36, but other versions should work as well. To set it up:

1. Access the psql interface and create a new database:
   ```
   sudo -u postgres psql
   create database chembl_36;
   ```

2. Download the `chembl_36_postgresql.tar.gz` file and extract it:
   ```
   tar xvzf chembl_36_postgresql.tar.gz
   ```

3. Restore the database (run from bash, not from inside psql):
   ```
   pg_restore --no-owner -U <user> --dbname=chembl_36 chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp --verbose
   ```

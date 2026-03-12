# timesplit-affinity-benchmark

Code for generating a time-split, novelty-filtered benchmark for protein-ligand binding affinity prediction.

## Installation

[uv](https://github.com/astral-sh/uv) is recommended. Requires Python 3.10+.

### uv (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### conda

```bash
conda create -n timesplit-affinity python=3.10
conda activate timesplit-affinity
pip install -e .
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

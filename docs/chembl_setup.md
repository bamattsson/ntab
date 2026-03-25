# Setting up ChEMBL

The benchmark pipeline requires a local ChEMBL PostgreSQL database. We used ChEMBL 36; other versions should work as well.

## 1. Create the database

Access the psql interface and create a new database:

```bash
sudo -u postgres psql
create database chembl_36;
```

## 2. Download the dump

Download [`chembl_36_postgresql.tar.gz`](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_postgresql.tar.gz) from the ChEMBL FTP server and extract it:

```bash
tar xvzf chembl_36_postgresql.tar.gz
```

## 3. Restore the database

Run from bash (not from inside psql):

```bash
pg_restore --no-owner -U <user> --dbname=chembl_36 chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp --verbose
```

from timesplit_affinity_benchmark.chembl_tools.chembl_requester import ChEMBLRequester


def test_get_chembl_id_to_smiles(mocker):
    """get_chembl_id_to_smiles should return whatever the cursor fetchall returns."""
    expected = [("CHEMBL1", "CCO"), ("CHEMBL2", "CCC")]

    mock_cursor = mocker.MagicMock()
    mock_cursor.fetchall.return_value = expected

    mock_conn = mocker.MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mocker.patch("psycopg2.connect", return_value=mock_conn)

    requester = ChEMBLRequester(host="localhost", user="user", password="pw", dbname="chembl_36")
    result = requester.get_chembl_id_to_smiles()

    assert result == expected
    mock_cursor.execute.assert_called_once()

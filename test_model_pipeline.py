"""Exemple de test unitaire."""

from model_pipeline import prepare_data


def test_prepare_data_returns_four_parts():
    result = prepare_data()
    assert len(result) == 4

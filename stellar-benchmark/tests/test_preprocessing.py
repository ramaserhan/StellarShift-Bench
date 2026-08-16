import pandas as pd

from stellar_benchmark.data.preprocessing import train_test_split_by_group


def test_group_split_prevents_star_leakage_and_is_deterministic():
    df = pd.DataFrame(
        {
            "source_id": ["a", "a", "b", "b", "c", "d", "e", "f"],
            "value": range(8),
        },
        index=[11, 12, 21, 22, 31, 41, 51, 61],
    )
    train_a, test_a = train_test_split_by_group(df, test_fraction=0.34, random_state=7)
    train_b, test_b = train_test_split_by_group(df, test_fraction=0.34, random_state=7)

    assert set(train_a.source_id).isdisjoint(set(test_a.source_id))
    assert train_a.equals(train_b)
    assert test_a.equals(test_b)
    assert len(train_a) + len(test_a) == len(df)


def test_group_split_rejects_missing_identifiers():
    df = pd.DataFrame({"source_id": ["a", None], "value": [1, 2]})
    try:
        train_test_split_by_group(df)
        assert False, "expected ValueError"
    except ValueError:
        pass

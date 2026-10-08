"""Token-capped batch pieces: every row once, order kept, cap respected, no piece left empty."""

from s1.train_decoder import split_by_tokens


def test_pieces_respect_cap_and_keep_every_row_in_order():
    lengths = [100, 900, 950, 1000, 120, 2000, 50, 400]
    idx = [3, 1, 2, 0, 5, 4, 7, 6]
    pieces = split_by_tokens(idx, lengths, max_tokens=2048)
    assert [i for p in pieces for i in p] == idx
    assert all(p for p in pieces)
    assert all(len(p) * max(lengths[i] for i in p) <= 2048 for p in pieces)


def test_small_batch_is_not_split_and_oversized_single_row_stands_alone():
    lengths = [10, 20, 30, 5000]
    assert split_by_tokens([0, 1, 2], lengths, 2048) == [[0, 1, 2]]
    assert split_by_tokens([0, 3, 1], lengths, 2048) == [[0], [3], [1]]

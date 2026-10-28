import pytest

from ragkit.chunking import chunk_words


def test_empty_text_yields_no_chunks():
    assert chunk_words("d", "   ") == []


def test_short_text_is_one_chunk():
    chunks = chunk_words("d", "one two three", size=10, overlap=2)
    assert [c.text for c in chunks] == ["one two three"]
    assert chunks[0].chunk_id == "d#0"


def test_windows_overlap_and_cover_everything():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_words("d", text, size=4, overlap=1)
    assert [c.text for c in chunks] == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_no_trailing_duplicate_window():
    chunks = chunk_words("d", " ".join(str(i) for i in range(8)), size=4, overlap=0)
    assert len(chunks) == 2


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (4, 4), (4, -1)])
def test_invalid_arguments_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_words("d", "a b c", size=size, overlap=overlap)

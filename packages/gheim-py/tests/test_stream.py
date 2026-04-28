"""Streaming deanonymizer is the load-bearing piece — exhaustive tests."""
from gheim.core.stream import StreamDeanonymizer

MAPPING = {"<PERSON_1>": "Joel", "<EMAIL_1>": "joel@example.com", "<PERSON_2>": "Alice"}


def _drive(chunks):
    """Feed each chunk into a fresh deanonymizer and return the concatenated output."""
    d = StreamDeanonymizer(MAPPING)
    out = "".join(d.transform(chunks))
    return out


def test_no_sentinel_passthrough():
    assert _drive(["Hello, world!"]) == "Hello, world!"


def test_single_chunk_one_sentinel():
    assert _drive(["Hi <PERSON_1>!"]) == "Hi Joel!"


def test_sentinel_split_across_two_chunks():
    assert _drive(["Hi <PER", "SON_1>!"]) == "Hi Joel!"


def test_sentinel_split_character_by_character():
    text = "Hello <PERSON_1>, your email is <EMAIL_1>."
    chunks = list(text)
    expected = "Hello Joel, your email is joel@example.com."
    assert _drive(chunks) == expected


def test_only_left_angle_delivered_then_unrelated():
    # '<' followed by something that is definitely not a sentinel — must emit '<' and the rest.
    assert _drive(["a < b and c < d"]) == "a < b and c < d"


def test_html_like_passthrough():
    assert _drive(["<html><body>hi</body></html>"]) == "<html><body>hi</body></html>"


def test_unknown_sentinel_passes_through_verbatim():
    assert _drive(["Code <SECRET_99> here"]) == "Code <SECRET_99> here"


def test_mixed_known_and_unknown():
    assert (
        _drive(["From <PERSON_1> to <SECRET_99>"]) == "From Joel to <SECRET_99>"
    )


def test_two_sentinels_back_to_back():
    assert _drive(["<PERSON_1><PERSON_2>"]) == "JoelAlice"


def test_two_sentinels_split_in_the_middle():
    # Boundary lands inside the second sentinel.
    assert _drive(["<PERSON_1><PER", "SON_2>"]) == "JoelAlice"


def test_left_angle_at_chunk_end_then_text_in_next():
    # '<' is held back; next chunk reveals it's not a sentinel.
    assert _drive(["foo <", " bar"]) == "foo < bar"


def test_left_angle_at_end_of_stream_emits_in_flush():
    # Stream ends mid-prefix — flush must emit the held bytes.
    assert _drive(["truncated <PERSON_"]) == "truncated <PERSON_"


def test_aggressive_chunking_pathological():
    text = "<PERSON_1> wrote to <PERSON_2> at <EMAIL_1>."
    expected = "Joel wrote to Alice at joel@example.com."
    # Try every possible single-split point.
    for i in range(len(text) + 1):
        chunks = [text[:i], text[i:]]
        assert _drive(chunks) == expected, f"failed at split={i}"


def test_three_way_chunking_at_every_boundary():
    text = "Hi <PERSON_1>!"
    expected = "Hi Joel!"
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            chunks = [text[:i], text[i:j], text[j:]]
            assert _drive(chunks) == expected, f"failed at i={i}, j={j}"


def test_empty_chunks_are_ignored():
    assert _drive(["", "<PERSON_1>", "", "!"]) == "Joel!"


def test_long_text_with_many_sentinels():
    parts = ["intro "] + ["<PERSON_1> " for _ in range(50)] + ["end"]
    out = _drive(parts)
    assert out == "intro " + "Joel " * 50 + "end"


def test_lowercase_sentinel_is_not_substituted():
    # Detector / model should have produced uppercase sentinels. Lowercase is
    # treated as ordinary text — passes through.
    assert _drive(["<person_1>"]) == "<person_1>"


def test_sentinel_inside_markdown_emphasis():
    assert _drive(["**<PERSON_1>**"]) == "**Joel**"


def test_feed_returns_only_safe_prefix():
    """A chunk ending with a sentinel-prefix should produce safe output but hold the prefix."""
    d = StreamDeanonymizer(MAPPING)
    out1 = d.feed("Hi <PER")
    assert out1 == "Hi "  # '<PER' is held back
    out2 = d.feed("SON_1>!")
    assert out2 == "Joel!"
    assert d.flush() == ""


def test_repeated_known_sentinel():
    assert _drive(["<PERSON_1> and <PERSON_1>"]) == "Joel and Joel"


def test_max_length_overflow_emits_verbatim():
    """A '<' followed by a very long valid-prefix-shape that never closes must eventually flush."""
    # '<' + 80 'A's — exceeds the max sentinel length, so the leading '<' should emit.
    long = "<" + "A" * 80
    out = _drive([long])
    assert out == long

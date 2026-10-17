import csv
import pathlib

import pytest

from bntts.bnnorm import cardinal, normalise, year

ROWS = list(csv.DictReader(open(pathlib.Path(__file__).parents[1] / "data/testset.tsv", encoding="utf-8"), delimiter="\t"))


def test_testset_shape():
    assert len(ROWS) == 50
    assert len({r["id"] for r in ROWS}) == 50
    cats = {r["category"] for r in ROWS}
    assert cats == {"news", "conversational", "numbers", "conjuncts", "banglish"}


@pytest.mark.parametrize("row", [r for r in ROWS if r["category"] != "banglish"], ids=lambda r: r["id"])
def test_normaliser_reproduces_reference(row):
    assert normalise(row["text"]) == row["spoken"]


@pytest.mark.parametrize("n,words", [(0, "শূন্য"), (28, "আটাশ"), (100, "একশো"), (1250, "এক হাজার দুইশো পঞ্চাশ"),
                                     (25000, "পঁচিশ হাজার"), (150000, "এক লাখ পঞ্চাশ হাজার"), (20000000, "দুই কোটি")])
def test_cardinal(n, words):
    assert cardinal(n) == words


def test_year():
    assert year(1971) == "উনিশশো একাত্তর"
    assert year(2026) == "দুই হাজার ছাব্বিশ"


def test_asr_style_digits_match_reference():
    assert normalise("16 ডিসেম্বর 1971 সালে") == "ষোলো ডিসেম্বর উনিশশো একাত্তর সালে"

from bntts.score import best_cer, cer


def test_identical_is_zero():
    assert cer("আজ ঢাকায় বৃষ্টি হতে পারে।", "আজ ঢাকায় বৃষ্টি হতে পারে") == 0.0


def test_dhaka_thaka_is_one_char():
    ref = "ঢাকা"
    assert cer(ref, "ঠাকা") == 0.25  # 4 code points, 1 substitution


def test_digits_vs_words_equal_after_normalising():
    assert cer("তাপমাত্রা আটাশ ডিগ্রি", "তাপমাত্রা ২৮ ডিগ্রি") == 0.0
    assert cer("তাপমাত্রা আটাশ ডিগ্রি", "তাপমাত্রা ২৮ ডিগ্রি", spoken=False) > 0


def test_codemix_accepts_either_script():
    refs = ["meeting টা কাল সকালে reschedule করো।", "মিটিং টা কাল সকালে রিশিডিউল করো।"]
    assert best_cer(refs, "Meeting টা কাল সকালে reschedule করো") == 0.0
    assert best_cer(refs, "মিটিংটা কাল সকালে রিশিডিউল করো") == 0.0

"""Core invariants.

Most of these target *silent* failure classes: slot misalignment, mask leakage,
position bias. Those produce wrong results without crashing, which is exactly
why they need to be pinned by assertions.
"""

from __future__ import annotations

import pytest
import torch

from jevlike.calibrate import fit_temperature, softmax_rows
from jevlike.data.build import subsample_options
from jevlike.data.registry import QSpec
from jevlike.metrics import evaluate_probs, expected_calibration_error
from jevlike.prompt import LETTERS, SENTINEL, build_prompt, render_questions
from jevlike.schema import MAX_OPTIONS, Answer, Question, SchemaError
from jevlike.scorer import NEG_INF, mask_to_options


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


# --- schema ------------------------------------------------------------------


def test_empty_and_single_option_rejected():
    with pytest.raises(SchemaError):
        Question.choice("q", "soru", ["tek"])
    with pytest.raises(SchemaError):
        Question.choice("", "soru", ["a", "b"])


def test_too_many_options_rejected():
    with pytest.raises(SchemaError):
        Question.choice("q", "soru", [f"o{i}" for i in range(MAX_OPTIONS + 1)])


def test_duplicate_options_rejected():
    with pytest.raises(SchemaError):
        Question.choice("q", "soru", ["a", "a"])


def test_score_expected_value():
    q = Question.score("p", "kac yildiz?", lo=1, hi=5)
    a = Answer(id="p", kind="score", options=q.options, probs=(0.0, 0.0, 0.0, 0.5, 0.5))
    assert a.expected_value(q.values) == pytest.approx(4.5)
    assert a.best == "4"


def test_bool_p_true_is_second_option():
    q = Question.boolean("b", "oyle mi?")
    a = Answer(id="b", kind="bool", options=q.options, probs=(0.3, 0.7))
    assert q.options[1] == "evet"
    assert a.as_dict()["p_true"] == pytest.approx(0.7)


# --- prompt / slot alignment ----------------------------------------------


def test_slot_token_ends_with_paren(tokenizer):
    """The most critical invariant: slot logits must predict the option letter."""
    qs = [
        Question.choice("a", "hangisi?", ["x", "y", "z"]),
        Question.boolean("b", "oyle mi?"),
        Question.score("c", "kac?", lo=1, hi=5),
    ]
    built = build_prompt(tokenizer, {"durum": "bir sey"}, qs)
    assert len(built.slot_positions) == 3
    for pos in built.slot_positions:
        tok = tokenizer.decode([built.input_ids[pos]])
        assert tok.endswith("("), f"slot token does not end in '(': {tok!r}"


def test_token_after_slot_is_sentinel(tokenizer):
    """The sentinel must not be a real answer; the model should read it as unanswered."""
    qs = [Question.choice("a", "hangisi?", ["x", "y"])]
    built = build_prompt(tokenizer, "durum", qs)
    pos = built.slot_positions[0]
    nxt = tokenizer.decode([built.input_ids[pos + 1]])
    assert nxt.startswith(SENTINEL)
    assert SENTINEL not in LETTERS


def test_option_counts_match_questions(tokenizer):
    qs = [
        Question.choice("a", "q", ["1", "2", "3", "4"]),
        Question.boolean("b", "q"),
    ]
    built = build_prompt(tokenizer, "s", qs)
    assert built.option_counts == (4, 2)
    assert built.question_ids == ("a", "b")


def test_slots_scale_with_question_count(tokenizer):
    """In a multi-question prompt every question needs its own slot, in one pass."""
    for k in (1, 2, 5, 12):
        qs = [Question.boolean(f"q{i}", f"soru {i}?") for i in range(k)]
        built = build_prompt(tokenizer, "durum", qs)
        assert len(built.slot_positions) == k
        assert len(set(built.slot_positions)) == k  # no collisions
        assert list(built.slot_positions) == sorted(built.slot_positions)


def test_render_assigns_letters_in_order():
    qs = [Question.choice("a", "soru", ["ilk", "ikinci", "ucuncu"])]
    text = render_questions(qs)
    assert "(A) ilk" in text
    assert "(B) ikinci" in text
    assert "(C) ucuncu" in text


# --- masking -------------------------------------------------------------


def test_invalid_letters_are_masked():
    logits = torch.zeros(1, 2, 52)
    counts = torch.tensor([[3, 2]])
    masked = mask_to_options(logits, counts)
    assert (masked[0, 0, :3] == 0).all()
    assert (masked[0, 0, 3:] == NEG_INF).all()
    assert (masked[0, 1, :2] == 0).all()
    assert (masked[0, 1, 2:] == NEG_INF).all()


def test_probability_sums_to_one_over_valid_options():
    logits = torch.randn(2, 3, 52)
    counts = torch.tensor([[4, 2, 0], [52, 5, 3]])
    probs = torch.softmax(mask_to_options(logits, counts), dim=-1)
    for i in range(2):
        for k in range(3):
            n = int(counts[i, k])
            if n == 0:
                continue
            assert probs[i, k, :n].sum().item() == pytest.approx(1.0, abs=1e-3)
            assert probs[i, k, n:].sum().item() == pytest.approx(0.0, abs=1e-3)


# --- metrics / calibration ----------------------------------------------


def test_ece_near_zero_when_perfectly_calibrated():
    """ECE should be near zero when confidence equals accuracy.

    With small n, equal-mass bins produce finite-sample noise (at n=200, bins of
    13 give 6/13 vs 7/13 -> ECE ~0.025) and the test starts measuring the sample
    rather than the code. Hence the large n.
    """
    n = 3000
    probs = [[0.5, 0.5] for _ in range(n)]
    labels = [i % 2 for i in range(n)]
    rep = evaluate_probs(probs, labels)
    assert rep.accuracy == pytest.approx(0.5)
    assert rep.mean_confidence == pytest.approx(0.5)
    assert rep.ece < 0.01
    assert abs(rep.overconfidence) < 0.01


def test_overconfidence_is_detected():
    probs = [[0.99, 0.01] for _ in range(100)]
    labels = [0] * 50 + [1] * 50  # true accuracy is 50%
    rep = evaluate_probs(probs, labels)
    assert rep.accuracy == pytest.approx(0.5)
    assert rep.overconfidence > 0.4
    assert rep.ece > 0.4


def test_equal_mass_bins_produce_no_empty_bins():
    conf = torch.rand(97).numpy()
    correct = (torch.rand(97) > 0.5).float().numpy()
    ece, mce = expected_calibration_error(conf, correct, n_bins=15)
    assert 0.0 <= ece <= 1.0
    assert mce >= ece


def test_temperature_fixes_overconfidence():
    """Fitting T must lower NLL and leave accuracy unchanged."""
    g = torch.Generator().manual_seed(0)
    n = 600
    labels = torch.randint(0, 4, (n,), generator=g).tolist()
    # genuinely ~60% correct, but with very sharp logits
    logits = []
    for i, y in enumerate(labels):
        row = torch.randn(4, generator=g).tolist()
        if i % 10 < 6:
            row[y] += 8.0
        else:
            row[(y + 1) % 4] += 8.0
        logits.append(row)

    cal = fit_temperature(logits, labels)
    assert cal.nll_after <= cal.nll_before + 1e-6
    assert cal.ece_after < cal.ece_before
    assert cal.temperature > 1.0  # softening expected

    before = evaluate_probs(softmax_rows(logits, 1.0), labels)
    after = evaluate_probs(softmax_rows(logits, cal.temperature), labels)
    assert before.accuracy == pytest.approx(after.accuracy)  # argmax is invariant


# --- data augmentation ----------------------------------------------------------


def test_menu_subsampling_keeps_gold_answer():
    import random

    rng = random.Random(0)
    options = [f"sinif_{i}" for i in range(77)]
    for label in (0, 13, 76):
        q = QSpec(id="q", prompt="p", kind="choice", options=list(options), label=label)
        gold = q.options[q.label]
        for _ in range(50):
            out = subsample_options(q, rng)
            assert out.options[out.label] == gold
            assert 2 <= len(out.options) <= MAX_OPTIONS
            assert len(set(out.options)) == len(out.options)


def test_menu_subsampling_has_no_position_bias():
    """The gold answer must not pile up on one letter."""
    import random
    from collections import Counter

    rng = random.Random(1)
    options = [f"s{i}" for i in range(8)]
    q = QSpec(id="q", prompt="p", kind="choice", options=options, label=3)
    positions = Counter(subsample_options(q, rng).label for _ in range(4000))
    # allow position 0 at most ~2x the expected rate (variable menu size makes
    # small indices naturally more frequent)
    assert positions[0] < 4000 * 0.5


def test_score_menus_are_not_shuffled():
    """Shuffling ordered levels would make expected value meaningless."""
    import random

    rng = random.Random(0)
    q = QSpec(
        id="p", prompt="kac?", kind="score", options=["1", "2", "3", "4", "5"],
        label=2, values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    out = subsample_options(q, rng)
    assert out.options == q.options
    assert out.label == q.label


def test_bool_menus_are_not_shuffled():
    """The options[1] == "true" convention must not be broken."""
    import random

    rng = random.Random(0)
    q = QSpec(id="b", prompt="oyle mi?", kind="bool", options=["hayir", "evet"], label=1)
    out = subsample_options(q, rng)
    assert out.options == ["hayir", "evet"]
    assert out.label == 1

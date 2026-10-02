"""Registry of open-source decision datasets.

All are public on the HF Hub and script-free (the datasets v3+ parquet path).
No distillation: labels are the datasets' own human annotations.

Two things are deliberate:

1. The `group` field. Datasets from the same task family share a group, and some
   groups are held out of training entirely (`HELD_OUT`). That is what makes
   "how does it do on a task type it has never seen" measurable -- training and
   evaluating on the same family would leak.

2. Multi-question sources. civil_comments / go_emotions / toxic-chat attach
   several labels to the same text; these become multi-slot examples in a single
   prompt. Needed so the model sees the multi-question format with real labels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class QSpec:
    """One question inside a record, plus its gold answer."""

    id: str
    prompt: str
    kind: str  # choice | bool | score
    options: list[str]
    label: int
    values: list[float] | None = None


@dataclass
class Record:
    """A state and one or more questions asked about it."""

    state: Any
    questions: list[QSpec]
    source: str
    group: str
    lang: str = "en"


@dataclass
class TaskSpec:
    name: str
    hf_id: str
    group: str
    convert: Callable[[Any, dict], Iterable[Record]]
    """(dataset, example) -> records. The dataset is passed to resolve label names."""
    config: str | None = None
    split: str = "train"
    eval_split: str | None = None
    lang: str = "en"
    max_train: int = 4000
    max_eval: int = 400
    notes: str = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def class_names(ds: Any, column: str) -> list[str] | None:
    feat = ds.features.get(column)
    names = getattr(feat, "names", None)
    return list(names) if names else None


def _clip(text: Any, limit: int = 1500) -> str:
    s = str(text).strip()
    return s if len(s) <= limit else s[:limit] + " …"


def simple_choice(
    text_cols: str | Sequence[str],
    label_col: str,
    prompt: str,
    *,
    qid: str = "label",
    state_keys: Sequence[str] | None = None,
    label_map: dict[str, str] | None = None,
) -> Callable[[Any, dict], Iterable[Record]]:
    """Converter for datasets with a single text column and a ClassLabel/str label."""
    cols = [text_cols] if isinstance(text_cols, str) else list(text_cols)
    keys = list(state_keys) if state_keys else cols
    # ds.unique() scans the whole column; calling it per example would read the
    # dataset N times. Compute once per dataset object and cache.
    options_cache: dict[int, list[str]] = {}

    def convert(ds: Any, ex: dict) -> Iterable[Record]:
        names = class_names(ds, label_col)
        raw = ex[label_col]
        if names is not None:
            options = list(names)
            label = int(raw)
        else:
            key = id(ds)
            if key not in options_cache:
                options_cache[key] = sorted({str(v) for v in ds.unique(label_col)})
            options = options_cache[key]
            value = str(raw)
            if value not in options:
                return []
            label = options.index(value)
        if label_map:
            options = [label_map.get(o, o) for o in options]

        state = {k: _clip(ex[c]) for k, c in zip(keys, cols)}
        if len(state) == 1:
            state = next(iter(state.values()))
        return [
            Record(
                state=state,
                questions=[QSpec(id=qid, prompt=prompt, kind="choice", options=options, label=label)],
                source="",
                group="",
            )
        ]

    return convert


def simple_score(
    text_col: str,
    label_col: str,
    prompt: str,
    values: Sequence[float],
    *,
    qid: str = "score",
    levels: Sequence[str] | None = None,
) -> Callable[[Any, dict], Iterable[Record]]:
    """For datasets with ordered levels (stars / ratings)."""
    lv = [str(v) for v in (levels if levels is not None else values)]

    def convert(ds: Any, ex: dict) -> Iterable[Record]:
        raw = ex[label_col]
        label = int(raw) if not isinstance(raw, str) else lv.index(raw.strip())
        return [
            Record(
                state=_clip(ex[text_col]),
                questions=[
                    QSpec(
                        id=qid,
                        prompt=prompt,
                        kind="score",
                        options=list(lv),
                        label=label,
                        values=[float(v) for v in values],
                    )
                ],
                source="",
                group="",
            )
        ]

    return convert


def mc_options(
    stem_col: str, choices_col: str, answer_col: str, prompt: str, *, qid: str = "answer"
) -> Callable[[Any, dict], Iterable[Record]]:
    """Multiple-choice datasets that carry their own option menu per example.

    These are the most valuable source for menu generalisation: the options
    differ on every example, so the menu cannot be memorised.
    """

    def convert(ds: Any, ex: dict) -> Iterable[Record]:
        ch = ex[choices_col]
        texts = list(ch["text"])
        labels = [str(x) for x in ch["label"]]
        key = str(ex[answer_col]).strip()
        if key not in labels:
            return []
        return [
            Record(
                state=_clip(ex[stem_col]),
                questions=[
                    QSpec(
                        id=qid, prompt=prompt, kind="choice", options=texts, label=labels.index(key)
                    )
                ],
                source="",
                group="",
            )
        ]

    return convert


# ---------------------------------------------------------------------------
# multi-question sources
# ---------------------------------------------------------------------------

# NOTE ON LANGUAGE: question prompts and bool labels below are Turkish. They are
# part of the trained interface -- the released adapter was trained with these
# exact strings, so translating them would silently degrade the model. Keeping
# the question language distinct from the (mostly English) data is also
# deliberate: it forces the model to treat options as arbitrary strings rather
# than memorised labels.

CIVIL_ASPECTS = [
    ("toksik", "toxicity", "Bu yorum toksik mi?"),
    ("agir_toksik", "severe_toxicity", "Bu yorum agir derecede toksik mi?"),
    ("mustehcen", "obscene", "Bu yorum mustehcen mi?"),
    ("tehdit", "threat", "Bu yorum tehdit iceriyor mu?"),
    ("hakaret", "insult", "Bu yorum hakaret iceriyor mu?"),
    ("kimlik_saldirisi", "identity_attack", "Bu yorum bir kimlik grubuna saldiri mi?"),
    ("cinsel_icerik", "sexual_explicit", "Bu yorum acik cinsel icerik barindiriyor mu?"),
]


#: How many moderation aspects to ask per example (see GO_EMOTIONS_PER_EXAMPLE).
CIVIL_ASPECTS_PER_EXAMPLE = 3


def convert_civil_comments(ds: Any, ex: dict) -> Iterable[Record]:
    """Several moderation booleans over the same text -> one prompt, many slots.

    Labels are annotator fractions (0..1). We drop the ambiguous band (0.2-0.8):
    forcing those into a binary label would inject noise and wreck calibration.
    """
    import random

    qs: list[QSpec] = []
    for qid, col, prompt in CIVIL_ASPECTS:
        v = float(ex[col])
        if 0.2 < v < 0.8:
            continue
        qs.append(
            QSpec(id=qid, prompt=prompt, kind="bool", options=["hayir", "evet"], label=int(v >= 0.8))
        )
    if not qs:
        return []
    if len(qs) > CIVIL_ASPECTS_PER_EXAMPLE:
        rng = random.Random(_clip(ex["text"], 64))
        qs = rng.sample(qs, k=CIVIL_ASPECTS_PER_EXAMPLE)
    return [Record(state=_clip(ex["text"]), questions=qs, source="", group="")]


def convert_toxic_chat(ds: Any, ex: dict) -> Iterable[Record]:
    """Toxicity + jailbreak on a user message: two decisions over the same text."""
    if not ex.get("human_annotation", False):
        return []
    return [
        Record(
            state={"kullanici_mesaji": _clip(ex["user_input"])},
            questions=[
                QSpec(
                    id="toksik",
                    prompt="Kullanici mesaji toksik mi?",
                    kind="bool",
                    options=["hayir", "evet"],
                    label=int(ex["toxicity"]),
                ),
                QSpec(
                    id="jailbreak",
                    prompt="Kullanici modelin kurallarini asmaya mi calisiyor?",
                    kind="bool",
                    options=["hayir", "evet"],
                    label=int(ex["jailbreaking"]),
                ),
            ],
            source="",
            group="",
        )
    ]


GO_EMOTION_SUBSET = [
    "admiration", "anger", "annoyance", "approval", "confusion", "curiosity",
    "disappointment", "disgust", "excitement", "fear", "gratitude", "joy",
    "love", "sadness", "surprise", "neutral",
]


#: How many emotions to ask about per example.
#
#: Because the loss is computed PER SLOT, expanding this dataset into 16
#: questions per example made it drive 27% of the gradient from 2500 examples
#: (2.6% of the data) -- wildly disproportionate. Asking a small random subset
#: per example fixes the balance and also varies which emotion is asked, so the
#: model cannot memorise a fixed question list.
GO_EMOTIONS_PER_EXAMPLE = 4


def convert_go_emotions(ds: Any, ex: dict) -> Iterable[Record]:
    """Multi-label emotion -> a few bool slots per example.

    The subset is drawn with a seed derived from the example's own id, so a
    rebuild is reproducible while still varying across examples.
    """
    import random

    names = class_names(ds, "labels") or []
    if not names:
        feat = ds.features["labels"]
        names = list(getattr(feat.feature, "names", []))
    present = {names[i] for i in ex["labels"] if i < len(names)}

    rng = random.Random(str(ex.get("id", ex["text"][:64])))
    asked = rng.sample(GO_EMOTION_SUBSET, k=min(GO_EMOTIONS_PER_EXAMPLE, len(GO_EMOTION_SUBSET)))
    qs = [
        QSpec(
            id=e,
            prompt=f"Bu metin '{e}' duygusunu ifade ediyor mu?",
            kind="bool",
            options=["hayir", "evet"],
            label=int(e in present),
        )
        for e in asked
    ]
    return [Record(state=_clip(ex["text"]), questions=qs, source="", group="")]


def convert_hellaswag(ds: Any, ex: dict) -> Iterable[Record]:
    endings = list(ex["endings"])
    try:
        label = int(ex["label"])
    except (TypeError, ValueError):
        return []
    return [
        Record(
            state={"baglam": _clip(ex["ctx"]), "etkinlik": ex.get("activity_label", "")},
            questions=[
                QSpec(
                    id="devam",
                    prompt="Baglam en dogal hangi sekilde devam eder?",
                    kind="choice",
                    options=endings,
                    label=label,
                )
            ],
            source="",
            group="",
        )
    ]


def convert_vitaminc(ds: Any, ex: dict) -> Iterable[Record]:
    options = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    lab = str(ex["label"]).strip().upper()
    if lab not in options:
        return []
    return [
        Record(
            state={"kanit": _clip(ex["evidence"]), "iddia": _clip(ex["claim"])},
            questions=[
                QSpec(
                    id="dogrulama",
                    prompt="Kanit iddiayi destekliyor mu, curutuyor mu, yoksa yetersiz mi?",
                    kind="choice",
                    options=options,
                    label=options.index(lab),
                )
            ],
            source="",
            group="",
        )
    ]


def convert_climate_fever(ds: Any, ex: dict) -> Iterable[Record]:
    names = class_names(ds, "claim_label") or []
    if not names:
        return []
    return [
        Record(
            state={"iddia": _clip(ex["claim"])},
            questions=[
                QSpec(
                    id="iklim_iddiasi",
                    prompt="Bu iklim iddiasinin bilimsel literaturdeki durumu nedir?",
                    kind="choice",
                    options=list(names),
                    label=int(ex["claim_label"]),
                )
            ],
            source="",
            group="",
        )
    ]


def convert_pair(
    a: str, b: str, label_col: str, prompt: str, keys: tuple[str, str], qid: str
) -> Callable[[Any, dict], Iterable[Record]]:
    def convert(ds: Any, ex: dict) -> Iterable[Record]:
        names = class_names(ds, label_col)
        if names is None:
            return []
        label = int(ex[label_col])
        if label < 0:
            return []
        return [
            Record(
                state={keys[0]: _clip(ex[a]), keys[1]: _clip(ex[b])},
                questions=[
                    QSpec(id=qid, prompt=prompt, kind="choice", options=list(names), label=label)
                ],
                source="",
                group="",
            )
        ]

    return convert


def convert_finance_options(ds: Any, ex: dict) -> Iterable[Record]:
    opts = list(ex["options"])
    gold = int(ex["gold_index"])
    if not (0 <= gold < len(opts)):
        return []
    return [
        Record(
            state=_clip(ex["input"]),
            questions=[
                QSpec(
                    id="finans_duygu",
                    prompt="Verilen secenekler arasindan dogru olani sec.",
                    kind="choice",
                    options=opts,
                    label=gold,
                )
            ],
            source="",
            group="",
        )
    ]


#: TRSAv1's "score" column is not stars but three ordered levels.
TRSA_LEVELS = ["Negative", "Neutral", "Positive"]


def convert_trsav1(ds: Any, ex: dict) -> Iterable[Record]:
    """Turkish product review -> three ordered levels (score kind).

    Score rather than choice because the levels are ordered: expected value is
    meaningful (-1 negative .. +1 positive), and a mistake between adjacent
    levels costs less Brier than one between distant levels.
    """
    level = str(ex["score"]).strip()
    if level not in TRSA_LEVELS:
        return []
    return [
        Record(
            state=_clip(ex["review"]),
            questions=[
                QSpec(
                    id="duygu_puani",
                    prompt="Bu urun yorumunun duygu yonu nedir?",
                    kind="score",
                    options=list(TRSA_LEVELS),
                    label=TRSA_LEVELS.index(level),
                    values=[-1.0, 0.0, 1.0],
                )
            ],
            source="",
            group="",
            lang="tr",
        )
    ]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

#: Groups never used in training. Generalisation is measured on these.
HELD_OUT: frozenset[str] = frozenset(
    {"commonsense", "iklim", "haber_grubu", "senaryo_tr", "guvenlik_sohbet", "finans"}
)

REGISTRY: list[TaskSpec] = [
    # --- intent / routing (high cardinality) ---
    TaskSpec(
        "banking77", "legacy-datasets/banking77", "niyet",
        simple_choice("text", "label", "Bu bankacilik talebi hangi niyete karsilik geliyor?", qid="niyet"),
        eval_split="test", max_train=6000,
        notes="77 classes -> ideal for menu subsampling",
    ),
    TaskSpec(
        "clinc150", "clinc/clinc_oos", "niyet",
        simple_choice("text", "intent", "Bu kullanici ifadesi hangi niyete karsilik geliyor?", qid="niyet"),
        config="plus", eval_split="validation", max_train=6000,
        notes="151 classes (including out-of-scope)",
    ),
    TaskSpec(
        "massive_intent_en", "mteb/amazon_massive_intent", "niyet",
        simple_choice("text", "label_text", "Bu sesli asistan komutu hangi niyete karsilik geliyor?", qid="niyet"),
        config="en", eval_split="test", max_train=3000,
    ),
    TaskSpec(
        "massive_intent_tr", "mteb/amazon_massive_intent", "niyet",
        simple_choice("text", "label_text", "Bu sesli asistan komutu hangi niyete karsilik geliyor?", qid="niyet"),
        config="tr", eval_split="test", lang="tr", max_train=3000,
    ),
    TaskSpec(
        "massive_scenario_tr", "mteb/amazon_massive_scenario", "senaryo_tr",
        simple_choice("text", "label_text", "Bu komut hangi alanla ilgili?", qid="senaryo"),
        config="tr", eval_split="test", lang="tr", max_eval=600,
        notes="HELD-OUT",
    ),
    # --- topic ---
    TaskSpec(
        "ag_news", "fancyzhx/ag_news", "konu",
        simple_choice("text", "label", "Bu haber hangi kategoriye ait?", qid="kategori"),
        eval_split="test",
    ),
    TaskSpec(
        "dbpedia14", "fancyzhx/dbpedia_14", "konu",
        simple_choice(["title", "content"], "label", "Bu metin hangi varlik turunu anlatiyor?",
                      qid="kategori", state_keys=["baslik", "icerik"]),
        eval_split="test",
    ),
    TaskSpec(
        "yahoo_topics", "community-datasets/yahoo_answers_topics", "konu",
        simple_choice(["question_title", "question_content"], "topic",
                      "Bu soru hangi konu basligina girer?", qid="konu",
                      state_keys=["baslik", "govde"]),
        eval_split="test", max_train=3000,
    ),
    TaskSpec(
        "newsgroups20", "SetFit/20_newsgroups", "haber_grubu",
        simple_choice("text", "label_text", "Bu gonderi hangi haber grubuna ait?", qid="grup"),
        eval_split="test", max_eval=600,
        notes="HELD-OUT",
    ),
    # --- emotion / score ---
    TaskSpec(
        "sst2", "stanfordnlp/sst2", "duygu",
        simple_choice("sentence", "label", "Bu cumlenin duygusu nedir?", qid="duygu"),
        eval_split="validation",
    ),
    TaskSpec(
        "sst5", "SetFit/sst5", "duygu",
        simple_choice("text", "label_text", "Bu cumlenin duygu yogunlugu nedir?", qid="duygu"),
        eval_split="test",
    ),
    TaskSpec(
        "yelp_full", "Yelp/yelp_review_full", "duygu",
        simple_score("text", "label", "Bu inceleme kac yildiz verir?", [1, 2, 3, 4, 5]),
        eval_split="test", max_train=3000,
    ),
    TaskSpec(
        "emotion6", "dair-ai/emotion", "duygu",
        simple_choice("text", "label", "Bu metindeki baskin duygu nedir?", qid="duygu"),
        eval_split="test",
    ),
    TaskSpec(
        "tweet_emotion", "cardiffnlp/tweet_eval", "duygu",
        simple_choice("text", "label", "Bu tweet'teki baskin duygu nedir?", qid="duygu"),
        config="emotion", eval_split="test",
    ),
    TaskSpec(
        "go_emotions", "google-research-datasets/go_emotions", "duygu",
        convert_go_emotions, config="simplified", eval_split="test", max_train=2500,
        notes=f"cok soruluk kaynak: ornek basina {GO_EMOTIONS_PER_EXAMPLE} bool yuvasi",
    ),
    # --- Turkish ---
    TaskSpec(
        "tr_urun_yorum", "asparius/Turkish-Product-Review", "duygu_tr",
        simple_choice("text", "label", "Bu urun yorumu olumlu mu olumsuz mu?", qid="duygu",
                      label_map={"0": "olumsuz", "1": "olumlu"}),
        eval_split="test", lang="tr",
    ),
    TaskSpec(
        "tr_sentiment", "winvoker/turkish-sentiment-analysis-dataset", "duygu_tr",
        simple_choice("text", "label", "Bu metnin duygusu nedir?", qid="duygu"),
        eval_split="test", lang="tr", max_train=3000,
    ),
    TaskSpec(
        "tr_trsav1", "maydogan/TRSAv1", "duygu_tr",
        convert_trsav1, lang="tr", max_train=2500,
    ),
    # --- NLI / verification ---
    TaskSpec(
        "mnli", "nyu-mll/glue", "nli",
        convert_pair("premise", "hypothesis", "label",
                     "Onculden hipotez cikarilabiliyor mu?", ("oncul", "hipotez"), "nli"),
        config="mnli", eval_split="validation_matched", max_train=4000,
    ),
    TaskSpec(
        "qnli", "nyu-mll/glue", "nli",
        convert_pair("question", "sentence", "label",
                     "Cumle soruyu cevapliyor mu?", ("soru", "cumle"), "nli"),
        config="qnli", eval_split="validation",
    ),
    TaskSpec(
        "rte", "nyu-mll/glue", "nli",
        convert_pair("sentence1", "sentence2", "label",
                     "Ilk cumleden ikincisi cikarilabiliyor mu?", ("cumle1", "cumle2"), "nli"),
        config="rte", eval_split="validation", max_train=2000,
    ),
    TaskSpec(
        "anli_r1", "facebook/anli", "nli",
        convert_pair("premise", "hypothesis", "label",
                     "Onculden hipotez cikarilabiliyor mu?", ("oncul", "hipotez"), "nli"),
        split="train_r1", eval_split="test_r1", max_train=2000,
    ),
    TaskSpec(
        "vitaminc", "tals/vitaminc", "dogrulama",
        convert_vitaminc, eval_split="test", max_train=3000,
    ),
    TaskSpec(
        "climate_fever", "tdiggelm/climate_fever", "iklim",
        convert_climate_fever, split="test", eval_split="test", max_eval=500,
        notes="HELD-OUT",
    ),
    # --- paraphrase ---
    TaskSpec(
        "mrpc", "nyu-mll/glue", "parafraz",
        convert_pair("sentence1", "sentence2", "label",
                     "Bu iki cumle ayni anlama mi geliyor?", ("cumle1", "cumle2"), "parafraz"),
        config="mrpc", eval_split="validation", max_train=2000,
    ),
    TaskSpec(
        "paws", "google-research-datasets/paws", "parafraz",
        convert_pair("sentence1", "sentence2", "label",
                     "Bu iki cumle parafraz mi?", ("cumle1", "cumle2"), "parafraz"),
        config="labeled_final", eval_split="test", max_train=2500,
    ),
    # --- moderation / safety ---
    TaskSpec(
        "civil_comments", "google/civil_comments", "moderasyon",
        convert_civil_comments, eval_split="test", max_train=2500,
        notes=f"cok soruluk kaynak: ornek basina <={CIVIL_ASPECTS_PER_EXAMPLE} bool yuvasi",
    ),
    TaskSpec(
        "toxic_conv", "SetFit/toxic_conversations", "moderasyon",
        simple_choice("text", "label_text", "Bu yorum toksik mi?", qid="toksik"),
        eval_split="test", max_train=2500,
    ),
    TaskSpec(
        "toxic_chat", "lmsys/toxic-chat", "guvenlik_sohbet",
        convert_toxic_chat, config="toxicchat0124", split="train", eval_split="test",
        max_eval=500, notes="HELD-OUT, cok soruluk",
    ),
    # --- spam ---
    TaskSpec(
        "sms_spam", "ucirvine/sms_spam", "spam",
        simple_choice("sms", "label", "Bu SMS spam mi?", qid="spam"),
        max_train=2000,
    ),
    TaskSpec(
        "enron_spam", "SetFit/enron_spam", "spam",
        simple_choice(["subject", "message"], "label_text", "Bu e-posta spam mi?",
                      qid="spam", state_keys=["konu", "govde"]),
        eval_split="test", max_train=2500,
    ),
    # --- per-example menus (menu generalisation) ---
    TaskSpec(
        "openbookqa", "allenai/openbookqa", "coktan_secmeli",
        mc_options("question_stem", "choices", "answerKey", "Asagidakilerden hangisi dogru?"),
        config="main", eval_split="test", max_train=2000,
    ),
    TaskSpec(
        "arc_easy", "allenai/ai2_arc", "coktan_secmeli",
        mc_options("question", "choices", "answerKey", "Bu sorunun dogru cevabi hangisi?"),
        config="ARC-Easy", eval_split="test", max_train=2000,
    ),
    TaskSpec(
        "arc_challenge", "allenai/ai2_arc", "coktan_secmeli",
        mc_options("question", "choices", "answerKey", "Bu sorunun dogru cevabi hangisi?"),
        config="ARC-Challenge", eval_split="test", max_train=1000,
    ),
    TaskSpec(
        "hellaswag", "Rowan/hellaswag", "coktan_secmeli",
        convert_hellaswag, eval_split="validation", max_train=2500,
    ),
    TaskSpec(
        "commonsense_qa", "tau/commonsense_qa", "commonsense",
        mc_options("question", "choices", "answerKey", "Bu sorunun dogru cevabi hangisi?"),
        eval_split="validation", max_eval=600,
        notes="HELD-OUT",
    ),
    TaskSpec(
        "finance_fpb", "AdaptLLM/finance-tasks", "finans",
        convert_finance_options, config="FPB", split="test", eval_split="test", max_eval=400,
        notes="HELD-OUT",
    ),
]


def held_out_groups() -> frozenset[str]:
    return HELD_OUT


def train_specs() -> list[TaskSpec]:
    return [s for s in REGISTRY if s.group not in HELD_OUT]


def eval_specs() -> list[TaskSpec]:
    """All specs: in-group ones are in-task, HELD_OUT ones measure generalisation."""
    return list(REGISTRY)


def by_name(name: str) -> TaskSpec:
    for s in REGISTRY:
        if s.name == name:
            return s
    raise KeyError(f"Unknown task: {name}")

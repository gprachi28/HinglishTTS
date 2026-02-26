# data/codeswitching.py
"""
Generate Hinglish sentences with word-level language tags assigned
AT GENERATION TIME — not detected after the fact since post-hoc 
language detection is fundamentally unreliable for ambiguous words

This avoids ambiguity with words like:
- 'is'       → Hindi (इस) or English verb?
- 'dunga'    → Hindi future tense, not in lookup
"""

import argparse
import csv
import logging
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Lang(str, Enum):
    HI = "HI"
    EN = "EN"


@dataclass
class TaggedToken:
    """A word with its language tag assigned at generation time."""
    word: str
    lang: Lang


@dataclass
class SwitchPoint:
    pattern: str
    tokens: List[TaggedToken]

    @property
    def sentence(self) -> str:
        return " ".join(t.word for t in self.tokens)

    @property
    def language_tags(self) -> List[str]:
        return [t.lang.value for t in self.tokens]

    @property
    def cmi(self) -> float:
        if not self.tokens:
            return 0.0
        en = sum(1 for t in self.tokens if t.lang == Lang.EN)
        hi = sum(1 for t in self.tokens if t.lang == Lang.HI)
        matrix = max(en, hi)
        return 1.0 - (matrix / len(self.tokens))

    @property
    def num_switches(self) -> int:
        switches = 0
        for i in range(1, len(self.tokens)):
            if self.tokens[i].lang != self.tokens[i-1].lang:
                switches += 1
        return switches


# ─────────────────────────────────────────────────────────────
# VOCABULARY BANKS — each word pre-tagged with its language
# ─────────────────────────────────────────────────────────────

def hi(*words) -> List[TaggedToken]:
    """Create Hindi tagged tokens."""
    result = []
    for word in words:
        result.extend(TaggedToken(w, Lang.HI) for w in word.split())
    return result


def en(*words) -> List[TaggedToken]:
    """Create English tagged tokens."""
    result = []
    for word in words:
        result.extend(TaggedToken(w, Lang.EN) for w in word.split())
    return result


# Hindi vocabulary banks — pre-tagged as HI
HI_SUBJECTS = [
    hi("main"), hi("hum"), hi("tum"), hi("aap"),
    hi("woh"), hi("mujhe"), hi("unhe"), hi("inhe"),
    hi("tumhara"), hi("humein"),
]

HI_VERBS = [
    hi("karna hai"), hi("karo"), hi("kar diya"),
    hi("ho gaya"), hi("dekh lena"), hi("bata do"),
    hi("samajh gaya"), hi("bol diya"), hi("sun lo"),
    hi("de do"), hi("le lena"), hi("aa jao"),
    hi("kar dunga"), hi("kar dungi"), hi("kar lena"),
    hi("bhool gaya"), hi("finish kar dunga"),
    hi("koshish kar raha hoon"),
]

HI_ADJECTIVES = [
    hi("bahut"), hi("thoda"), hi("zyada"),
    hi("bilkul"), hi("ekdum"), hi("achha"),
    hi("mushkil"), hi("aasaan"),
]

HI_TIME = [
    hi("kal"), hi("aaj"), hi("parso"), hi("abhi"),
    hi("baad mein"), hi("pehle"), hi("is hafte"),
    hi("agli baar"), hi("subah"), hi("shaam ko"),
    hi("raat ko"), hi("kal tak"),
]

HI_DISCOURSE = [
    hi("yaar"), hi("arre"), hi("dekho"), hi("suno"),
    hi("acha"), hi("theek hai"), hi("matlab"),
    hi("waise"), hi("toh"),
]

# Hindi sentence fragments — all pre-tagged
HI_MATRIX_FRAGMENTS = [
    hi("Yeh idea"), hi("Yeh project"),
    hi("Aaj ka"), hi("Kal ka"),
    hi("Is hafte ka"), hi("Yeh kaam"),
]

HI_CONNECTORS = [
    hi("ke baare mein baat karna chahta hoon"),
    hi("tak complete karna hai"),
    hi("dekh raha tha"),
    hi("bahut important hai"),
    hi("pe kaam kar raha hoon"),
    hi("submit karna bhool gaya"),
    hi("bahut helpful tha"),
    hi("bahut lamba tha"),
    hi("bahut lamba hoga"),
]

# English vocabulary banks — pre-tagged as EN
EN_NP = [
    en("meeting"), en("deadline"), en("project"),
    en("presentation"), en("report"), en("email"),
    en("feedback"), en("update"), en("call"),
    en("review"), en("proposal"), en("budget"),
    en("timeline"), en("team"), en("phone"),
    en("coffee"), en("lunch"), en("break"),
    en("plan"), en("idea"), en("problem"),
    en("solution"), en("decision"),
]

EN_VERBS = [
    en("cancel"), en("postpone"), en("reschedule"),
    en("confirm"), en("share"), en("send"),
    en("check"), en("fix"), en("manage"),
    en("handle"), en("discuss"), en("finalize"),
    en("approve"), en("submit"), en("review"),
]

EN_ADJECTIVES = [
    en("boring"), en("interesting"), en("difficult"),
    en("easy"), en("important"), en("urgent"),
    en("serious"), en("amazing"), en("terrible"),
    en("perfect"), en("helpful"), en("long"),
]

EN_MATRIX_FRAGMENTS = [
    en("Please send me the"),
    en("I will"),
    en("Can you"),
    en("The"),
    en("Let's discuss this"),
    en("We need to"),
    en("Make sure you"),
    en("I finished the"),
    en("I think"),
]

EN_CONNECTORS = [
    en("don't worry"),
    en("please check"),
    en("after the"),
    en("is really"),
    en("looks"),
    en("is due"),
    en("was"),
]


def r(bank: list):
    """Random choice from a bank."""
    return random.choice(bank)


# ─────────────────────────────────────────────────────────────
# SENTENCE BUILDERS
# Each builder returns List[TaggedToken] with correct tags
# ─────────────────────────────────────────────────────────────

def build_hindi_matrix_english_np() -> List[TaggedToken]:
    """Hindi sentence with English noun phrase."""
    patterns = [
        lambda: r(HI_MATRIX_FRAGMENTS) + r(EN_NP) + r(HI_CONNECTORS),
        lambda: r(HI_SUBJECTS) + r(EN_NP) + hi("ke baare mein") + r(HI_VERBS),
        lambda: hi("Yeh") + r(EN_NP) + r(HI_ADJECTIVES) + hi("important hai"),
        lambda: r(HI_SUBJECTS) + r(EN_NP) + r(HI_TIME) + en("send") + hi("kar dunga"),
        lambda: hi("Tumhara") + r(EN_NP) + r(HI_ADJECTIVES) + en("helpful") + hi("tha"),
    ]
    return random.choice(patterns)()


def build_hindi_matrix_english_verb() -> List[TaggedToken]:
    """Hindi sentence with English verb/adjective."""
    patterns = [
        lambda: hi("Yeh situation") + r(HI_ADJECTIVES) + en("complicated") + hi("ho gayi hai"),
        lambda: r(HI_SUBJECTS) + hi("yeh kaam") + r(EN_ADJECTIVES) + hi("tarike se") + r(HI_VERBS),
        lambda: hi("Meeting") + r(EN_VERBS) + hi("kar do,") + r(HI_TIME) + hi("nahi hogi"),
        lambda: r(HI_SUBJECTS) + hi("yeh file") + r(EN_VERBS) + hi("karna chahta hoon"),
        lambda: hi("Pehle") + r(EN_VERBS) + hi("karo, phir baat karte hain"),
        lambda: hi("Please yeh") + r(EN_VERBS) + hi("mat karo") + r(HI_TIME),
    ]
    return random.choice(patterns)()


def build_english_matrix_hindi_np() -> List[TaggedToken]:
    """English sentence with Hindi time/expression."""
    patterns = [
        lambda: en("Please send me the") + r(EN_NP) + r(HI_TIME),
        lambda: en("I will") + r(EN_VERBS) + en("it") + r(HI_TIME) + en(", don't worry"),
        lambda: en("Can you") + r(EN_VERBS) + en("this") + r(HI_TIME) + en("?"),
        lambda: en("The") + r(EN_NP) + en("is due") + r(HI_TIME) + en(", please check"),
        lambda: en("Let's discuss this") + r(HI_TIME) + en("after the") + r(EN_NP),
        lambda: en("We need to") + r(EN_VERBS) + en("this") + r(HI_TIME),
    ]
    return random.choice(patterns)()


def build_inter_sentential() -> List[TaggedToken]:
    """Language switch between clauses."""
    patterns = [
        lambda: r(HI_SUBJECTS) + hi("soch raha tha .") + en("The") + r(EN_NP) + en("is really") + r(EN_ADJECTIVES),
        lambda: hi("Yeh") + r(EN_NP) + r(HI_ADJECTIVES) + hi("important hai .") + en("Please") + r(EN_VERBS) + en("it") + r(HI_TIME),
        lambda: en("The deadline is") + r(HI_TIME) + en(".") + r(HI_SUBJECTS) + hi("ready hoon"),
        lambda: hi("Kal meeting hai .") + en("Make sure you") + r(EN_VERBS) + en("the") + r(EN_NP),
        lambda: en("I finished the") + r(EN_NP) + en(".") + r(HI_ADJECTIVES) + hi("acha laga"),
    ]
    return random.choice(patterns)()


def build_tag_switching() -> List[TaggedToken]:
    """Discourse marker switching."""
    patterns = [
        lambda: r(HI_DISCOURSE) + en(", the") + r(EN_NP) + en("is") + r(EN_ADJECTIVES),
        lambda: en("The meeting was long,") + r(HI_DISCOURSE),
        lambda: r(HI_DISCOURSE) + en(",") + r(HI_SUBJECTS) + hi("yeh") + r(EN_NP) + hi("finish kar dunga"),
        lambda: en("Yeh") + r(EN_ADJECTIVES) + en("hai,") + r(HI_DISCOURSE),
        lambda: r(HI_DISCOURSE) + en(", can you") + r(EN_VERBS) + en("this") + r(HI_TIME) + en("?"),
    ]
    return random.choice(patterns)()


BUILDERS = {
    "hindi_matrix_english_np":   build_hindi_matrix_english_np,
    "hindi_matrix_english_verb": build_hindi_matrix_english_verb,
    "english_matrix_hindi_np":   build_english_matrix_hindi_np,
    "inter_sentential":          build_inter_sentential,
    "tag_switching":             build_tag_switching,
}


def generate_sentences(
    num_sentences: int,
    min_cmi: float = 0.1,
    max_cmi: float = 0.7,
    seed: int = 42,
) -> List[SwitchPoint]:
    random.seed(seed)
    results = []
    attempts = 0
    max_attempts = num_sentences * 10

    while len(results) < num_sentences and attempts < max_attempts:
        attempts += 1

        pattern_name = random.choice(list(BUILDERS.keys()))
        try:
            tokens = BUILDERS[pattern_name]()
        except Exception as e:
            logger.warning(f"Builder failed: {e}")
            continue

        if len(tokens) < 4:
            continue

        sp = SwitchPoint(pattern=pattern_name, tokens=tokens)

        if not (min_cmi <= sp.cmi <= max_cmi):
            continue

        results.append(sp)

    logger.info(f"Generated {len(results)} sentences in {attempts} attempts")
    return results


def save_sentences(sentences: List[SwitchPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "pattern", "cmi", "num_switches", "language_tags"])
        for sp in sentences:
            writer.writerow([
                sp.sentence,
                sp.pattern,
                f"{sp.cmi:.3f}",
                sp.num_switches,
                " ".join(sp.language_tags),
            ])
    logger.info(f"Saved {len(sentences)} sentences to {output_path}")


def print_samples(sentences: List[SwitchPoint], n: int = 10) -> None:
    print("\n" + "="*60)
    print("SAMPLE HINGLISH SENTENCES WITH CORRECT LANGUAGE TAGS")
    print("="*60)
    for sp in random.sample(sentences, min(n, len(sentences))):
        print(f"\n[{sp.pattern}]")
        print(f"  Text   : {sp.sentence}")
        print(f"  Tags   : {' '.join(sp.language_tags)}")
        print(f"  CMI    : {sp.cmi:.3f} | Switches: {sp.num_switches}")
    print("="*60)


def main(output_path: str, num_sentences: int, seed: int) -> None:
    sentences = generate_sentences(num_sentences, seed=seed)
    print_samples(sentences)
    save_sentences(sentences, Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",    type=str, default="data/codeswitched/sentences.csv")
    parser.add_argument("--num_sentences",  type=int, default=5000)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    main(args.output_path, args.num_sentences, args.seed)

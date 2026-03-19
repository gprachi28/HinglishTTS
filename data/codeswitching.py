# data/codeswitching.py
"""
Generate Hinglish (Hindi-English) code-switched sentences for TTS.
Linguistically motivated using Equivalence Constraint theory.
All tags assigned at generation time — no post-hoc detection.

TAGGING RULE: Any English loanword inside a Hindi phrase must be
split: en("word") + hi("rest"). Never put English words inside hi().
"""

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Lang(str, Enum):
    HI = "HI"
    EN = "EN"


@dataclass
class TaggedToken:
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
        return 1.0 - (max(en, hi) / len(self.tokens))

    @property
    def num_switches(self) -> int:
        switches = 0
        for i in range(1, len(self.tokens)):
            if self.tokens[i].lang != self.tokens[i-1].lang:
                switches += 1
        return switches


def hi(*words) -> List[TaggedToken]:
    result = []
    for word in words:
        result.extend(TaggedToken(w, Lang.HI) for w in word.split())
    return result


def en(*words) -> List[TaggedToken]:
    result = []
    for word in words:
        result.extend(TaggedToken(w, Lang.EN) for w in word.split())
    return result


def r(bank): return random.choice(bank)


# ─────────────────────────────────────────────────────────────
# VOCABULARY BANKS
# All English loanwords split out into en() — never inside hi()
# ─────────────────────────────────────────────────────────────

# ── Hindi subjects ───────────────────────────────────────────
HI_SUBJECTS = [
    hi("main"), hi("hum"), hi("tum"), hi("aap"),
    hi("woh"), hi("mujhe"), hi("unhe"), hi("inhe"),
    hi("humein"),
]

HI_SUBJECTS_SOCIAL = [
    hi("main"), hi("hum"), hi("tum"), hi("aap"),
    hi("meri behen"), hi("mera bhai"),
    hi("mera dost"), hi("meri friend"),
]

HI_SUBJECTS_FIRST_PERSON = [
    hi("main"), hi("hum"),
]

# ── Hindi verbs ───────────────────────────────────────────────
HI_VERBS_GENERAL = [
    hi("kar diya"), hi("kar dunga"), hi("kar lena"),
    hi("ho gaya"), hi("ho jaega"), hi("ho sakta hai"),
    hi("dekh lena"), hi("dekh liya"),
    hi("bata do"), hi("bata diya"),
    hi("samajh gaya"), hi("samajh lo"),
    hi("sun lo"), hi("sun liya"),
    hi("de do"), hi("de diya"),
    hi("le lena"), hi("le liya"),
    hi("bhool gaya"), hi("yaad hai"),
    hi("pata hai"), hi("pata nahi"),
    hi("lag raha hai"), hi("lagta hai"),
    hi("koshish karunga"),
    # FIX: "try" is English
    en("try") + hi("karta hoon"),
]

HI_VERBS_FUTURE = [
    hi("kar dunga"), hi("kar lenge"),
    hi("ho jaega"), hi("mil jaega"),
    hi("bata dunga"), hi("dekh lunga"),
]

# ── Hindi intensifiers ────────────────────────────────────────
HI_INTENSIFIERS = [
    hi("bahut"), hi("thoda"), hi("zyada"),
    hi("bilkul"), hi("ekdum"), hi("kaafi"),
    hi("itna"), hi("thodi si"), hi("bahut zyada"),
]

# ── Hindi adjectives — pure Hindi only ───────────────────────
# FIX: "important", "interesting", "boring", "complicated" are English
# Moved to EN_ADJECTIVES — do not put them in HI_ADJECTIVES
HI_ADJECTIVES_PURE = [
    hi("achha"), hi("bura"), hi("mushkil"),
    hi("aasaan"), hi("ajeeb"), hi("sahi"),
    hi("galat"), hi("seedha"),
]

# ── Hindi time expressions ────────────────────────────────────
HI_TIME_SHORT = [
    hi("kal"), hi("aaj"), hi("abhi"),
    hi("parso"), hi("pehle"), hi("baad mein"),
]

HI_TIME_LONG = [
    hi("is hafte"), hi("agli baar"), hi("pichhli baar"),
    hi("aaj shaam ko"), hi("kal subah"),
    hi("do din mein"), hi("thodi der mein"),
    hi("raat ko"), hi("subah subah"),
    hi("jaldi se"),
]

HI_TIME = HI_TIME_SHORT + HI_TIME_LONG

# ── Hindi discourse markers ───────────────────────────────────
HI_DISCOURSE = [
    hi("yaar"), hi("arre"), hi("dekho"),
    hi("suno"), hi("acha"), hi("theek hai"),
    hi("matlab"), hi("waise"), hi("toh"),
    hi("haan"), hi("sach mein"),
    hi("ek kaam karo"), hi("sun na"),
    hi("bata na"),
]

# ── Hindi connectors ─────────────────────────────────────────
HI_CONNECTORS = [
    hi("ke baare mein baat karni hai"),
    hi("ke liye kaam kar raha hoon"),
    hi("pe dhyan dena"),
    hi("tak complete karna hai"),
    hi("mein problem aa rahi hai"),
    hi("ke baad free hoon"),
    hi("ki wajah se late ho gaya"),
    hi("ke saath koi issue hai"),
]

# ── Hindi emotional expressions ───────────────────────────────
# FIX: "tension" and "stress" are English loanwords — split out
HI_EMOTIONS = [
    hi("mujhe bohot achha laga"),
    hi("yeh sunke bura laga"),
    hi("main khush hoon"),
    hi("thak gaya hoon"),
    # FIX: en("tension") + hi(...)
    en("tension") + hi("ho rahi hai"),
    hi("maza aa gaya"),
    hi("pagal ho jaaunga"),
]

# ── English nouns — professional ──────────────────────────────
EN_NP_WORK = [
    en("meeting"), en("deadline"), en("project"),
    en("presentation"), en("report"), en("email"),
    en("feedback"), en("update"), en("follow-up"),
    en("review"), en("proposal"), en("budget"),
    en("timeline"), en("sprint"), en("roadmap"),
    en("client call"), en("status update"),
    en("pull request"), en("code review"),
    en("team standup"), en("one-on-one"),
]

EN_NP_EVERYDAY = [
    en("phone"), en("coffee"), en("lunch"),
    en("break"), en("weekend"), en("plan"),
    en("idea"), en("problem"), en("solution"),
    en("decision"), en("movie"), en("song"),
    en("gym"), en("diet"), en("routine"),
    en("party"), en("trip"), en("order"),
    en("delivery"), en("subscription"), en("password"),
]

EN_NP_TECH = [
    en("laptop"), en("charger"), en("wifi"),
    en("app"), en("update"), en("bug"),
    en("feature"), en("server"), en("database"),
    en("API"), en("deployment"), en("script"),
]

EN_NP_EDUCATION = [
    en("class"), en("assignment"), en("exam"),
    en("semester"), en("lecture"), en("internship"),
    en("project"), en("presentation"), en("campus"),
    en("college"), en("course"), en("degree"),
    en("scholarship"), en("tuition"), en("syllabus"),
]

EN_NP_ENTERTAINMENT = [
    en("movie"), en("series"), en("episode"),
    en("Netflix"), en("concert"), en("show"),
    en("reel"), en("podcast"), en("playlist"),
    en("trailer"), en("review"), en("ticket"),
    en("stream"), en("channel"), en("vlog"),
]

EN_NP_SOCIAL = [
    en("post"), en("story"), en("comment"),
    en("like"), en("follow"), en("vibe"),
    en("selfie"), en("reel"), en("caption"),
    en("group"), en("chat"), en("status"),
    en("meme"), en("feed"), en("profile"),
]

EN_NP_FOOD_LIFESTYLE = [
    en("burger"), en("pizza"), en("cafe"),
    en("brunch"), en("smoothie"), en("salad"),
    en("cheat day"), en("meal prep"), en("recipe"),
    en("diet"), en("detox"), en("snack"),
    en("dessert"), en("menu"), en("takeaway"),
]

EN_NP_ALL = EN_NP_WORK + EN_NP_EVERYDAY + EN_NP_TECH + EN_NP_EDUCATION + EN_NP_ENTERTAINMENT + EN_NP_SOCIAL + EN_NP_FOOD_LIFESTYLE

# General pool — excludes professional/work nouns
EN_NP_GENERAL = EN_NP_EVERYDAY + EN_NP_TECH + EN_NP_EDUCATION + EN_NP_ENTERTAINMENT + EN_NP_SOCIAL + EN_NP_FOOD_LIFESTYLE

# Countable only — for "I have a ..." templates
EN_NP_COUNTABLE = [
    en("meeting"), en("call"), en("deadline"),
    en("presentation"), en("review"), en("one-on-one"),
    en("client call"), en("team standup"), en("proposal"),
    en("idea"), en("plan"), en("problem"),
    en("solution"), en("trip"), en("delivery"),
    en("exam"), en("assignment"), en("class"),
    en("concert"), en("show"), en("ticket"),
]

# ── English verbs ─────────────────────────────────────────────
EN_VERBS_WORK = [
    en("cancel"), en("postpone"), en("reschedule"),
    en("confirm"), en("share"), en("send"),
    en("check"), en("fix"), en("manage"),
    en("handle"), en("discuss"), en("finalize"),
    en("approve"), en("submit"), en("review"),
    en("escalate"), en("prioritize"), en("delegate"),
]

EN_VERBS_EVERYDAY = [
    en("book"), en("order"), en("call"),
    en("message"), en("download"),
    en("install"), en("update"), en("reset"),
    en("charge"), en("connect"), en("upload"),
]

EN_VERBS_SOCIAL = [
    en("post"), en("share"), en("like"),
    en("follow"), en("unfollow"), en("comment"),
    en("stream"), en("watch"), en("subscribe"),
    en("google"), en("search"), en("save"),
]

EN_VERBS_ALL = EN_VERBS_WORK + EN_VERBS_EVERYDAY + EN_VERBS_SOCIAL

# Quick physical/task verbs that work with "karke aata hoon"
# (implies "I'll go do X and come back" — excludes abstract delegation verbs)
EN_VERBS_QUICK_TASK = [
    en("check"), en("fix"), en("reset"), en("update"),
    en("send"), en("share"), en("call"), en("message"),
    en("download"), en("install"), en("book"), en("order"),
    en("charge"), en("connect"),
]

# ── English adjectives ────────────────────────────────────────
EN_ADJECTIVES = [
    en("boring"), en("interesting"), en("difficult"),
    en("easy"), en("important"), en("urgent"),
    en("serious"), en("amazing"), en("terrible"),
    en("perfect"), en("helpful"), en("long"),
    en("short"), en("quick"), en("slow"),
    en("complicated"), en("simple"), en("random"),
    en("valid"), en("optional"),
]


# ─────────────────────────────────────────────────────────────
# SENTENCE BUILDERS
# ─────────────────────────────────────────────────────────────

def build_hindi_matrix_english_np() -> List[TaggedToken]:
    patterns = [
        # Work context
        lambda: hi("Aaj ka") + r(EN_NP_WORK) + hi("bahut lamba tha"),
        lambda: hi("Kal ka") + r(EN_NP_WORK) + r(HI_TIME_SHORT) + hi("hai"),
        lambda: hi("main") + r(EN_NP_WORK) + hi("pe kaam kar raha hoon"),
        lambda: hi("main") + r(EN_NP_WORK) + r(HI_TIME_SHORT) + hi("tak bhej dunga"),
        lambda: hi("Aaj ka") + r(EN_NP_WORK) + hi("cancel ho gaya"),
        lambda: hi("Mera") + r(EN_NP_WORK) + hi("abhi bhi pending hai"),

        # General context — uses broader pool
        lambda: hi("Yeh") + r(EN_NP_GENERAL) + hi("bahut") + r(EN_ADJECTIVES) + hi("lag raha hai"),
        lambda: hi("Is") + r(EN_NP_GENERAL) + hi("ki wajah se late ho gaya"),
        lambda: hi("Tumhara") + r(EN_NP_GENERAL) + hi("mujhe mil gaya"),
        lambda: hi("Mera") + r(EN_NP_GENERAL) + hi("khatam ho gaya"),
        lambda: hi("Yeh") + r(EN_NP_GENERAL) + hi("ka kya plan hai"),
        lambda: r(HI_SUBJECTS_SOCIAL) + r(EN_NP_GENERAL) + hi("lena bhool gaya"),
        lambda: hi("Aaj raat") + r(EN_NP_GENERAL) + hi("ka plan kya hai"),

        # Education context
        lambda: hi("Mera") + r(EN_NP_EDUCATION) + r(HI_TIME_SHORT) + hi("hai"),
        lambda: hi("Is") + r(EN_NP_EDUCATION) + hi("ke liye bilkul taiyar nahi hoon"),
        lambda: r(HI_SUBJECTS_SOCIAL) + r(EN_NP_EDUCATION) + hi("mein bahut achha hai"),

        # Entertainment context
        lambda: hi("Yeh") + r(EN_NP_ENTERTAINMENT) + r(HI_INTENSIFIERS) + en("good") + hi("tha"),
        lambda: r(HI_SUBJECTS_SOCIAL) + r(EN_NP_ENTERTAINMENT) + hi("dekha kya"),
        lambda: hi("Naya") + r(EN_NP_ENTERTAINMENT) + hi("aa gaya hai"),

        # Social media context
        lambda: hi("Tera") + r(EN_NP_SOCIAL) + r(HI_INTENSIFIERS) + en("viral") + hi("ho gaya"),
        lambda: r(HI_SUBJECTS_SOCIAL) + r(EN_NP_SOCIAL) + hi("dekh ke") + r(HI_INTENSIFIERS) + hi("khush ho gaya"),

        # Tech context
        lambda: hi("Mera") + r(EN_NP_TECH) + hi("kaam nahi kar raha"),
        lambda: hi("Is") + r(EN_NP_TECH) + hi("mein problem aa rahi hai"),
        lambda: hi("Naya") + r(EN_NP_TECH) + hi("aa gaya hai"),
    ]
    return r(patterns)()


def build_hindi_matrix_english_verb() -> List[TaggedToken]:
    patterns = [
        lambda: hi("Pehle") + r(EN_VERBS_ALL) + hi("karo, phir baat karte hain"),
        lambda: hi("Please yeh") + r(EN_VERBS_ALL) + hi("mat karo") + r(HI_TIME_SHORT),
        lambda: hi("main") + hi("yeh") + r(EN_VERBS_ALL) + hi("karna chahta hoon"),
        lambda: hi("Isko") + r(EN_VERBS_ALL) + hi("karna hai") + r(HI_TIME),
        lambda: hi("Jaldi se") + r(EN_VERBS_ALL) + hi("kar do"),
        lambda: hi("main") + hi("khud") + r(EN_VERBS_ALL) + hi("kar lunga"),
        lambda: hi("Yeh") + r(EN_VERBS_ALL) + hi("karna zaroori hai"),
        lambda: hi("main") + hi("pehle") + r(EN_VERBS_QUICK_TASK) + hi("karke aata hoon"),
        lambda: hi("Kya tum") + r(EN_VERBS_ALL) + hi("kar sakte ho"),
        lambda: hi("Yeh") + r(EN_VERBS_EVERYDAY + EN_VERBS_SOCIAL) + hi("karna itna mushkil nahi hai"),
        lambda: hi("Usne") + r(EN_VERBS_SOCIAL) + hi("kar diya, ab kya karein"),
        lambda: hi("main") + r(EN_VERBS_SOCIAL) + hi("karta hoon, tum dekho"),
    ]
    return r(patterns)()


def build_english_matrix_hindi_np() -> List[TaggedToken]:
    patterns = [
        lambda: en("Please send me the") + r(EN_NP_WORK) + r(HI_TIME_SHORT),
        lambda: en("Can you") + r(EN_VERBS_ALL) + en("this") + r(HI_TIME_SHORT) + en("?"),
        lambda: en("The") + r(EN_NP_WORK) + en("is due") + r(HI_TIME_SHORT) + en(", please check"),
        lambda: en("Let's discuss this") + r(HI_TIME) + en("after the") + r(EN_NP_WORK),
        lambda: en("We need to") + r(EN_VERBS_ALL) + en("this") + r(HI_TIME_SHORT),
        lambda: en("I'll") + r(EN_VERBS_ALL) + en("it") + r(HI_TIME_SHORT) + en(", don't worry"),
        lambda: en("Make sure you") + r(EN_VERBS_ALL) + en("the") + r(EN_NP_WORK) + r(HI_TIME_SHORT),
        lambda: en("The") + r(EN_NP_WORK) + en("got") + r(EN_VERBS_WORK) + r(HI_TIME_LONG),
        lambda: en("I was thinking we should") + r(EN_VERBS_ALL) + en("the") + r(EN_NP_EVERYDAY) + r(HI_TIME),
        lambda: en("Could you please") + r(EN_VERBS_ALL) + en("the") + r(EN_NP_WORK) + r(HI_TIME_SHORT) + hi("tak chahiye"),
    ]
    return r(patterns)()


def build_inter_sentential() -> List[TaggedToken]:
    patterns = [
        lambda: r(HI_SUBJECTS_FIRST_PERSON) + hi("soch raha tha.") + en("The") + r(EN_NP_ALL) + en("is really") + r(EN_ADJECTIVES),
        lambda: hi("Kal meeting hai.") + en("Make sure you") + r(EN_VERBS_ALL) + en("the") + r(EN_NP_ALL),
        lambda: hi("Yeh kaam ho gaya.") + en("Now let's focus on the") + r(EN_NP_ALL),
        # FIX: r(HI_EMOTIONS) already contains correctly tagged tokens
        lambda: r(HI_EMOTIONS) + en(". The") + r(EN_NP_EVERYDAY) + en("was really") + r(EN_ADJECTIVES),
        lambda: hi("Bahut thak gaya hoon.") + en("This") + r(EN_NP_EVERYDAY) + en("is not helping"),
        lambda: hi("Samajh nahi aa raha.") + en("The") + r(EN_NP_TECH) + en("keeps giving errors"),
        lambda: en("I finished the") + r(EN_NP_WORK) + en(".") + r(HI_INTENSIFIERS) + hi("acha laga"),
        lambda: en("The") + r(EN_NP_WORK) + en("got cancelled.") + r(HI_SUBJECTS_FIRST_PERSON) + hi("khush hoon"),
        # FIX: en("ready") + hi("hoon") — not hi("ready hoon")
        lambda: en("I have a") + r(EN_NP_COUNTABLE) + r(HI_TIME_SHORT) + en(".") + r(HI_SUBJECTS_FIRST_PERSON) + en("ready") + hi("hoon"),
        lambda: en("The") + r(EN_NP_TECH) + en("is not working.") + hi("Kya karein ab"),
        lambda: en("This") + r(EN_NP_EVERYDAY) + en("is") + r(EN_ADJECTIVES) + en(".") + hi("Mujhe nahi pasand"),
    ]
    return r(patterns)()


def build_tag_switching() -> List[TaggedToken]:
    patterns = [
        lambda: r(HI_DISCOURSE) + en(", the") + r(EN_NP_ALL) + en("is") + r(EN_ADJECTIVES),
        lambda: r(HI_DISCOURSE) + en(", main yeh") + r(EN_NP_ALL) + hi("finish kar dunga"),
        lambda: r(HI_DISCOURSE) + en(", can you") + r(EN_VERBS_ALL) + en("this") + r(HI_TIME_SHORT) + en("?"),
        lambda: en("The") + r(EN_NP_ALL) + en("was") + r(EN_ADJECTIVES) + en(",") + r(HI_DISCOURSE),
        lambda: r(HI_DISCOURSE) + en(", this") + r(EN_NP_ALL) + en("needs to be") + r(EN_VERBS_WORK) + r(HI_TIME_SHORT),
        lambda: r(HI_DISCOURSE) + en(", I think the") + r(EN_NP_WORK) + en("is") + r(EN_ADJECTIVES),
        # FIX: en("stress") not inside hi()
        lambda: en("Honestly,") + hi("main yeh") + r(EN_NP_ALL) + hi("se thak gaya hoon"),
        lambda: r(HI_DISCOURSE) + en(", let's just") + r(EN_VERBS_ALL) + en("it") + r(HI_TIME_SHORT),
    ]
    return r(patterns)()


def build_everyday_conversation() -> List[TaggedToken]:
    patterns = [
        # Food and social
        lambda: r(HI_SUBJECTS_SOCIAL) + en("lunch") + hi("ke liye kahan chalein") + r(HI_TIME_SHORT),
        lambda: hi("Aaj") + en("coffee") + hi("peeni hai,") + r(HI_SUBJECTS_SOCIAL) + hi("aa raha hai kya"),
        lambda: hi("Kal") + en("party") + hi("mein") + r(HI_INTENSIFIERS) + en("fun") + hi("tha"),
        lambda: hi("Yaar,") + en("this restaurant") + hi("ka khana") + r(HI_INTENSIFIERS) + en("good") + hi("hai"),
        # FIX: hi("Main") not hi("mujhe"), en("diet") not inside hi()
        lambda: hi("Main") + en("diet") + hi("pe hoon, isliye") + en("dessert") + hi("nahi lunga"),

        # Travel
        lambda: hi("Aaj") + en("traffic") + r(HI_INTENSIFIERS) + en("bad") + hi("tha"),
        lambda: r(HI_SUBJECTS_SOCIAL) + en("cab") + hi("book kar liya,") + r(HI_TIME_SHORT) + hi("aa jaega"),
        lambda: hi("Kal") + en("flight") + hi("hai,") + r(HI_SUBJECTS_SOCIAL) + hi("packing nahi ki abhi tak"),

        # Health
        lambda: r(HI_SUBJECTS_SOCIAL) + en("gym") + hi("jaana") + r(HI_INTENSIFIERS) + hi("mushkil ho gaya hai"),
        # FIX: en("tough") not inside hi()
        lambda: hi("Yeh") + en("workout") + r(HI_INTENSIFIERS) + en("tough") + hi("tha"),
        lambda: r(HI_SUBJECTS_SOCIAL) + en("doctor") + hi("se milna hai") + r(HI_TIME),

        # Tech
        lambda: r(HI_SUBJECTS_SOCIAL) + en("phone") + hi("ki") + en("battery") + r(HI_INTENSIFIERS) + hi("jaldi khatam hoti hai"),
        # FIX: en("slow") not inside hi()
        lambda: hi("Mera") + en("laptop") + en("slow") + hi("ho gaya hai"),
        lambda: r(HI_SUBJECTS_SOCIAL) + en("password") + hi("bhool gaya, ab kya karoon"),
        # FIX: en("useful") not inside hi()
        lambda: hi("Yeh naya") + en("app") + r(HI_INTENSIFIERS) + en("useful") + hi("hai"),

        # Shopping
        lambda: r(HI_SUBJECTS_SOCIAL) + en("online") + hi("order kiya tha,") + en("delivery") + r(HI_TIME_SHORT) + hi("aani chahiye"),
        # FIX: en("good deals") not inside hi()
        lambda: hi("Yeh") + en("sale") + hi("mein") + r(HI_INTENSIFIERS) + en("good deals") + hi("the"),
        # FIX: en("size") not inside hi()
        lambda: r(HI_SUBJECTS_SOCIAL) + en("return") + hi("karna hai yeh,") + en("size") + hi("sahi nahi hai"),

        # Work from home
        lambda: r(HI_SUBJECTS_FIRST_PERSON) + en("work from home") + hi("kar raha hoon") + r(HI_TIME),
        # FIX: en("internet"), en("slow") not inside hi()
        lambda: hi("Aaj") + en("internet") + r(HI_INTENSIFIERS) + en("slow") + hi("hai, kaam nahi ho raha"),
        # FIX: en("zoom call"), en("connection") not inside hi()
        lambda: r(HI_SUBJECTS_FIRST_PERSON) + en("zoom call") + hi("pe tha,") + en("connection") + hi("baar baar jaata tha"),
    ]
    return r(patterns)()


def build_emotional_expression() -> List[TaggedToken]:
    patterns = [
        lambda: r(HI_EMOTIONS) + en("after the") + r(EN_NP_EVERYDAY),
        lambda: r(HI_DISCOURSE) + en(", I can't believe this") + r(EN_NP_EVERYDAY) + hi("itna lamba tha"),
        lambda: hi("Sach mein,") + en("this") + r(EN_NP_EVERYDAY) + hi("ne mujhe") + r(HI_INTENSIFIERS) + hi("khush kar diya"),
        # FIX: en("shocked") not inside hi()
        lambda: r(HI_SUBJECTS_SOCIAL) + hi("yeh sunke") + r(HI_INTENSIFIERS) + en("shocked") + hi("ho gaya"),
        # FIX: en("stress") not inside hi()
        lambda: hi("Itna") + en("stress") + hi("hai") + r(HI_TIME_LONG) + hi("se"),
        lambda: hi("Sach keh raha hoon,") + en("this") + r(EN_NP_EVERYDAY) + hi("best hai"),
        lambda: r(HI_SUBJECTS_SOCIAL) + r(EN_NP_EVERYDAY) + hi("dekh ke") + r(HI_INTENSIFIERS) + hi("khush ho gaya"),
        lambda: r(HI_EMOTIONS) + en(". I didn't expect the") + r(EN_NP_EVERYDAY) + hi("itna achha hoga"),
    ]
    return r(patterns)()


BUILDERS = {
    "hindi_matrix_english_np":   build_hindi_matrix_english_np,
    "hindi_matrix_english_verb": build_hindi_matrix_english_verb,
    "english_matrix_hindi_np":   build_english_matrix_hindi_np,
    "inter_sentential":          build_inter_sentential,
    "tag_switching":             build_tag_switching,
    "everyday_conversation":     build_everyday_conversation,
    "emotional_expression":      build_emotional_expression,
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
        pattern_name = r(list(BUILDERS.keys()))
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


def print_samples(sentences: List[SwitchPoint], n: int = 15) -> None:
    print("\n" + "="*60)
    print("SAMPLE HINGLISH SENTENCES")
    print("="*60)
    for sp in random.sample(sentences, min(n, len(sentences))):
        print(f"\n[{sp.pattern}]")
        print(f"  Text : {sp.sentence}")
        print(f"  Tags : {' '.join(sp.language_tags)}")
    print("="*60)


def main(output_path: str, num_sentences: int, seed: int) -> None:
    sentences = generate_sentences(num_sentences, seed=seed)
    print_samples(sentences)
    save_sentences(sentences, Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",   type=str, default="data/codeswitched/sentences.csv")
    parser.add_argument("--num_sentences", type=int, default=5000)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    main(args.output_path, args.num_sentences, args.seed)
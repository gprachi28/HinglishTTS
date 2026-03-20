# data/vocabulary.py
"""
Vocabulary banks for Hinglish code-switched sentence generation.

Contains all tagged word/phrase pools used by the sentence builders
in codeswitching.py. Each entry is a list of TaggedToken objects
created via hi() / en() helpers.

TAGGING RULE: Any English loanword must use en(), even inside a
Hindi phrase. Never put English words inside hi().
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
import random


class Lang(str, Enum):
    HI = "HI"
    EN = "EN"


@dataclass
class TaggedToken:
    word: str
    lang: Lang


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
# HINDI VOCABULARY BANKS
# ─────────────────────────────────────────────────────────────

# ── Subjects ─────────────────────────────────────────────────
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

# ── Verbs ────────────────────────────────────────────────────
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
    # "try" is English
    en("try") + hi("karta hoon"),
]

HI_VERBS_FUTURE = [
    hi("kar dunga"), hi("kar lenge"),
    hi("ho jaega"), hi("mil jaega"),
    hi("bata dunga"), hi("dekh lunga"),
]

# ── Intensifiers ─────────────────────────────────────────────
HI_INTENSIFIERS = [
    hi("bahut"), hi("thoda"), hi("zyada"),
    hi("bilkul"), hi("ekdum"), hi("kaafi"),
    hi("itna"), hi("thodi si"), hi("bahut zyada"),
]

# ── Adjectives — pure Hindi only ─────────────────────────────
HI_ADJECTIVES_PURE = [
    hi("achha"), hi("bura"), hi("mushkil"),
    hi("aasaan"), hi("ajeeb"), hi("sahi"),
    hi("galat"), hi("seedha"),
]

# ── Time expressions ─────────────────────────────────────────
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

# ── Discourse markers ────────────────────────────────────────
HI_DISCOURSE = [
    hi("yaar"), hi("arre"), hi("dekho"),
    hi("suno"), hi("acha"), hi("theek hai"),
    hi("matlab"), hi("waise"), hi("toh"),
    hi("haan"), hi("sach mein"),
    hi("ek kaam karo"), hi("sun na"),
    hi("bata na"),
]

# ── Connectors ───────────────────────────────────────────────
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

# ── Emotional expressions ────────────────────────────────────
# "tension" and "stress" are English loanwords — split out
HI_EMOTIONS = [
    hi("mujhe bohot achha laga"),
    hi("yeh sunke bura laga"),
    hi("main khush hoon"),
    hi("thak gaya hoon"),
    en("tension") + hi("ho rahi hai"),
    hi("maza aa gaya"),
    hi("pagal ho jaaunga"),
]


# ─────────────────────────────────────────────────────────────
# ENGLISH VOCABULARY BANKS
# ─────────────────────────────────────────────────────────────

# ── Nouns — professional ─────────────────────────────────────
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

# ── Composite noun pools ─────────────────────────────────────
EN_NP_ALL = (
    EN_NP_WORK + EN_NP_EVERYDAY + EN_NP_TECH + EN_NP_EDUCATION
    + EN_NP_ENTERTAINMENT + EN_NP_SOCIAL + EN_NP_FOOD_LIFESTYLE
)

# General pool — excludes professional/work nouns
EN_NP_GENERAL = (
    EN_NP_EVERYDAY + EN_NP_TECH + EN_NP_EDUCATION
    + EN_NP_ENTERTAINMENT + EN_NP_SOCIAL + EN_NP_FOOD_LIFESTYLE
)

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

# ── Verbs ────────────────────────────────────────────────────
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

# ── Adjectives ───────────────────────────────────────────────
EN_ADJECTIVES = [
    en("boring"), en("interesting"), en("difficult"),
    en("easy"), en("important"), en("urgent"),
    en("serious"), en("amazing"), en("terrible"),
    en("perfect"), en("helpful"), en("long"),
    en("short"), en("quick"), en("slow"),
    en("complicated"), en("simple"), en("random"),
    en("valid"), en("optional"),
]

# ── Numerical / Entity tokens (CS-06) ────────────────────────
EN_DAYS = [
    en("Monday"), en("Tuesday"), en("Wednesday"), en("Thursday"),
    en("Friday"), en("Saturday"), en("Sunday"),
    en("Monday morning"), en("Friday evening"),
    en("this Sunday"), en("next Monday"), en("next Friday"),
]

EN_TIMES = [
    en("9 AM"), en("10 AM"), en("11 AM"), en("12 PM"),
    en("1 PM"), en("2 PM"), en("3 PM"), en("4 PM"), en("5 PM"), en("6 PM"),
    en("9:30 AM"), en("10:30 AM"), en("2:30 PM"), en("4:30 PM"),
]

EN_DATES = [
    en("1st"), en("5th"), en("10th"), en("15th"), en("20th"), en("25th"), en("30th"),
    en("March 15"), en("April 1"), en("end of month"), en("end of week"),
]

EN_ENTITIES = [
    en("Google Meet"), en("Zoom"), en("WhatsApp"), en("Teams"),
    en("Gmail"), en("Slack"), en("Instagram"), en("LinkedIn"),
    en("Google Drive"), en("Notion"), en("Jira"),
]

# ── Intraword verb stems (CS-07) ─────────────────────────────
# English verb stems that take Hindi grammatical suffixes
EN_INTRAWORD_VERBS = [
    en("unfriend"), en("unfollow"), en("block"), en("mute"),
    en("screenshot"), en("forward"), en("react"), en("tag"),
    en("edit"), en("crop"), en("zoom"), en("scroll"),
    en("record"), en("upload"), en("download"), en("share"),
    en("save"), en("like"), en("repost"), en("reply"),
]

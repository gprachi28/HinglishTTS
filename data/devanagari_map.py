# data/devanagari_map.py
"""
Hand-curated Romanized Hindi → Devanagari mapping for the HinglishTTS vocabulary.

Only 182 unique Hindi-tagged words appear in the generated sentences.
A manual dictionary gives 100% accuracy — no neural model or transliteration
scheme can reliably handle informal romanization ("main" → "मैं", not "मैन").

This file is the single source of truth for Devanagari conversion.
To add new vocabulary to codeswitching.py, also add the mapping here.
"""

# fmt: off
ROMAN_TO_DEVANAGARI: dict[str, str] = {
    # ── Pronouns & Subjects ────────────────────────────────────
    "main":     "मैं",
    "maine":    "मैंने",
    "hum":      "हम",
    "tum":      "तुम",
    "aap":      "आप",
    "woh":      "वो",
    "mujhe":    "मुझे",
    "unhe":     "उन्हें",
    "inhe":     "इन्हें",
    "humein":   "हमें",
    "mera":     "मेरा",
    "meri":     "मेरी",
    "tera":     "तेरा",
    "tumhara":  "तुम्हारा",
    "uski":     "उसकी",
    "usne":     "उसने",
    "use":      "उसे",
    "yeh":      "यह",
    "usse":     "उससे",
    "tune":     "तूने",
    "unhone":   "उन्होंने",
    "is":       "इस",
    "isko":     "इसको",

    # ── Family / Social ────────────────────────────────────────
    "behen":    "बहन",
    "bhai":     "भाई",
    "dost":     "दोस्त",
    "friend":   "फ्रेंड",

    # ── Common Verbs ───────────────────────────────────────────
    "kar":      "कर",
    "karo":     "करो",
    "karna":    "करना",
    "karta":    "करता",
    "karte":    "करते",
    "karke":    "करके",
    "karein":   "करें",
    "karoon":   "करूं",
    "kiya":     "किया",
    "ho":       "हो",
    "hoon":     "हूँ",
    "hai":      "है",
    "hain":     "हैं",
    "tha":      "था",
    "the":      "थे",
    "hoga":     "होगा",
    "hoti":     "होती",
    "gaya":     "गया",
    "liya":     "लिया",
    "lena":     "लेना",
    "diya":     "दिया",
    "dena":     "देना",
    "de":       "दे",
    "do":       "दो",
    "dekh":     "देख",
    "dekho":    "देखो",
    "dekha":    "देखा",
    "sun":      "सुन",
    "sunke":    "सुनके",
    "suno":     "सुनो",
    "bata":     "बता",
    "baat":     "बात",
    "soch":     "सोच",
    "samajh":   "समझ",
    "bhool":    "भूल",
    "mil":      "मिल",
    "milna":    "मिलना",
    "milte":    "मिलते",
    "lag":      "लग",
    "laga":     "लगा",
    "raha":     "रहा",
    "rahi":     "रही",
    "sakte":    "सकते",
    "chahta":   "चाहता",
    "chahiye":  "चाहिए",
    "jaana":    "जाना",
    "jaata":    "जाता",
    "jaaunga":  "जाऊँगा",
    "jaega":    "जाएगा",
    "aata":     "आता",
    "aani":     "आनी",
    "aa":       "आ",
    "bhej":     "भेज",
    "lunga":    "लूँगा",
    "dunga":    "दूँगा",
    "padega":   "पड़ेगा",
    "peeni":    "पीनी",

    # ── Intensifiers ───────────────────────────────────────────
    "bahut":    "बहुत",
    "bohot":    "बहोत",
    "thoda":    "थोड़ा",
    "thodi":    "थोड़ी",
    "zyada":    "ज़्यादा",
    "bilkul":   "बिल्कुल",
    "ekdum":    "एकदम",
    "kaafi":    "काफ़ी",
    "itna":     "इतना",
    "si":       "सी",

    # ── Adjectives ─────────────────────────────────────────────
    "achha":    "अच्छा",
    "bura":     "बुरा",
    "mushkil":  "मुश्किल",
    "aasaan":   "आसान",
    "ajeeb":    "अजीब",
    "sahi":     "सही",
    "galat":    "गलत",
    "seedha":   "सीधा",
    "lamba":    "लंबा",
    "naya":     "नया",
    "best":     "बेस्ट",

    # ── Time Expressions ───────────────────────────────────────
    "kal":      "कल",
    "aaj":      "आज",
    "abhi":     "अभी",
    "parso":    "परसो",
    "pehle":    "पहले",
    "baad":     "बाद",
    "baar":     "बार",
    "hafte":    "हफ़्ते",
    "subah":    "सुबह",
    "shaam":    "शाम",
    "raat":     "रात",
    "din":      "दिन",
    "der":      "देर",
    "jaldi":    "जल्दी",
    "agli":     "अगली",
    "pichhli":  "पिछली",

    # ── Discourse Markers ──────────────────────────────────────
    "yaar":     "यार",
    "arre":     "अरे",
    "acha":     "अच्छा",
    "theek":    "ठीक",
    "matlab":   "मतलब",
    "waise":    "वैसे",
    "toh":      "तो",
    "haan":     "हाँ",
    "sach":     "सच",
    "na":       "ना",

    # ── Connectors / Postpositions ─────────────────────────────
    "ke":       "के",
    "ki":       "की",
    "ka":       "का",
    "ko":       "को",
    "se":       "से",
    "mein":     "में",
    "pe":       "पे",
    "tak":      "तक",
    "par":      "पर",
    "ne":       "ने",
    "wala":     "वाला",
    "wajah":    "वजह",
    "liye":     "लिए",
    "bhi":      "भी",
    "isliye":   "इसलिए",
    "baje":     "बजे",

    # ── Emotional / Misc ───────────────────────────────────────
    "khush":    "ख़ुश",
    "thak":     "थक",
    "pagal":    "पागल",
    "maza":     "मज़ा",
    "pasand":   "पसंद",
    "taiyar":   "तैयार",
    "zaroori":  "ज़रूरी",
    "late":     "लेट",

    # ── Action / State Words ───────────────────────────────────
    "kaam":     "काम",
    "khana":    "खाना",
    "khatam":   "ख़त्म",
    "keh":      "कह",
    "khud":     "ख़ुद",
    "kya":      "क्या",
    "kahan":    "कहाँ",
    "nahi":     "नहीं",
    "mat":      "मत",
    "ek":       "एक",
    "sab":      "सब",
    "phir":     "फिर",
    "ab":       "अब",

    # ── Domain-Specific Hindi ──────────────────────────────────
    "chalein":  "चलें",
    "chhutti":  "छुट्टी",
    "packing":  "पैकिंग",
    "bacha":    "बचा",
    "tarikh":   "तारीख़",
    "slot":     "स्लॉट",
    "available":"अवेलेबल",
    "pending":  "पेंडिंग",
    "shift":    "शिफ़्ट",
    "plan":     "प्लान",
    "please":   "प्लीज़",

    # ── English loanwords tagged as HI (appear in HI context) ──
    "meeting":  "मीटिंग",
    "cancel":   "कैंसल",
    "order":    "ऑर्डर",
    "book":     "बुक",
    "problem":  "प्रॉब्लम",
    "post":     "पोस्ट",
    "payment":  "पेमेंट",
    "submit":   "सबमिट",
    "reschedule": "रिशेड्यूल",
    "finish":   "फ़िनिश",
    "deadline": "डेडलाइन",
    "interview":"इंटरव्यू",
}
# fmt: on


def transliterate_hindi(word: str) -> str:
    """Look up Devanagari for a Romanized Hindi word.

    Falls back to the original word (in Roman) if not in the dictionary.
    Handles attached punctuation (e.g. "hai," → "है,").
    """
    PUNCT = ".,?!;:"
    suffix = ""
    clean = word
    while clean and clean[-1] in PUNCT:
        suffix = clean[-1] + suffix
        clean = clean[:-1]
    if not clean:
        return suffix

    result = ROMAN_TO_DEVANAGARI.get(clean.lower(), clean)
    return result + suffix

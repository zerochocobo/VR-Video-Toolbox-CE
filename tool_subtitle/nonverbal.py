"""Classify a transcribed line as pure non-verbal vocalisation.

``HAS_LINGUISTIC_CONTENT_RE`` in :mod:`tool_subtitle.logic` only asks whether a
line contains *any* kana, kanji or letter, so a moan transcribed as ``あぁん``
passes it whole. For subtitles that is a cosmetic problem. For dubbing it is
not: :mod:`tool_clonevoice_v2` hands every surviving line to IndexTTS, so the
moan is synthesised and spoken over the picture.

The test here is the two-layer one from WhisperJAV
(``modules/subtitle_pipeline/cleaners/nonlinguistic_utterance_filter.py``):

1. **Evidence check.** Kanji, particles, verb endings, question markers or any
   of a list of common words mean this is dialogue -- keep it, no further
   questions asked.
2. **Sound-kana check.** Only if no evidence was found: every remaining kana
   must belong to a defined sound alphabet (breathing はひふへほ, kissing
   ちぢつづ, nasal んッー, ...). All of them, or the line is kept.

The ordering is what makes it safe. Nearly every sound kana is also a real
Japanese syllable, so the alphabet on its own would delete ``ふるえる`` and
``おっぱい``; it only ever runs on lines that failed to show a single sign of
grammar. WhisperJAV reports 302 of 2414 lines matched on a real title with no
real-dialogue hits.

Japanese only, by construction -- the evidence list *is* Japanese grammar. A
line with no Japanese at all is never classified as non-verbal here, so other
source languages fall through untouched.
"""
from __future__ import annotations

import re

# --- character classes ------------------------------------------------------

_HIRAGANA = r"ぁ-ゟ"
_KATAKANA = r"゠-ヿ"
_KANJI = r"一-鿿㐀-䶿豈-﫿"

_JAPANESE_RE = re.compile(f"[{_HIRAGANA}{_KATAKANA}]")
_KANJI_RE = re.compile(f"[{_KANJI}]")

# Sound-effect alphabet. Grouped by the vocalisation each row spells out, so a
# future addition lands next to its neighbours rather than at the end.
SOUND_KANA: frozenset[str] = frozenset(
    # vowels and small vowels -- the core of moaning and gasping
    "あぁいぃうぅえぇおぉ"
    "アァイィウゥエェオォ"
    # nasal, geminate, prolongation
    "んンっッーゝゞ"
    # breathing
    "はひふへほハヒフヘホ"
    # kissing and sucking
    "ちぢつづチヂツヅ"
    # slurping and sighing
    "すずスズ"
    # swallowing
    "くぐこごクグコゴ"
    # licking
    "ぺべれめペベレメ"
    # popping and puffing
    "ぷぱぶばプパブバ"
    # nasal closure
    "むぬムヌ"
    # rolling liquids
    "ろるロル"
    # voiced fricative
    "じジ"
    # small ya/yu/yo
    "ゃゅょャュョ"
)

# Stripped before the sound-kana check so punctuation never keeps a line alive.
PUNCTUATION_CHARS: frozenset[str] = frozenset(
    " \t　"
    "、。，．,.!?！？"
    "…‥・~〜"
    "「」『』【】()（）<>〈〉《》"
    "\"'“”‘’«»"
    "─—–-"
    ":;：；"
    "♪♡♥★☆●○◎"
    "*＊"
)

# --- language evidence ------------------------------------------------------

_MULTI_CHAR_PARTICLES = [
    "から", "まで", "より", "ほど", "くらい", "ぐらい",
    "だけ", "しか", "ばかり", "なんて", "なんか", "って",
    "では", "には", "とは", "とも", "への", "での",
]

_SINGLE_CHAR_PARTICLES = "をがでとものにへ"

_VERB_ENDINGS = [
    "ます", "ません", "ました", "ましょう",
    "です", "でした", "でしょう",
    "ない", "なかった", "なくて",
    "たい", "たくない", "たがる",
    "ちゃう", "じゃう", "てしまう", "でしまう",
    "てる", "でる", "ている", "でいる",
    "ちゃった", "じゃった",
    "なきゃ", "なくちゃ", "なければ",
    "られる", "させる", "される",
    "だろう", "だった",
    "ておく", "てあげる", "てくれる", "てもらう",
]

_QUESTION_MARKERS = [
    "の？", "のか", "ですか", "ますか", "ましたか",
    "だっけ", "っけ", "かな", "かしら",
]

# Words that are themselves spelled entirely in sound kana carry their weight
# twice here: they are evidence of dialogue, and without them the sound-kana
# check would delete them (ふるえる, おっぱい, ぐちゃぐちゃ).
_COMMON_WORDS = [
    "お願い", "すみません", "ごめん", "ありがとう", "どうぞ",
    "はい", "いいえ", "こんにちは", "こんばんは", "おはよう",
    "あれ", "これ", "それ", "どれ",
    "ここ", "そこ", "あそこ", "どこ",
    "こう", "そう", "ああ", "どう",
    "なん", "なに", "だれ", "いつ", "なぜ", "どうして",
    "私", "わたし", "あたし", "あなた", "おまえ", "お前",
    "君", "きみ", "彼", "彼女", "みんな", "皆",
    "僕", "ぼく", "俺", "おれ", "あいつ", "こいつ",
    "先生", "せんせい", "先輩", "せんぱい",
    "お兄", "お姉", "おにい", "おねえ",
    "いい", "よい", "悪い", "わるい",
    "やばい", "すごい", "凄い",
    "可愛い", "かわいい", "綺麗", "きれい",
    "美しい", "うつくしい", "素晴らしい",
    "ひどい", "おかしい", "嬉しい", "うれしい",
    "痛い", "いたい", "熱い", "あつい",
    "冷たい", "つめたい", "温かい", "あたたかい",
    "気持ちいい", "きもちいい",
    "大きい", "おおきい", "小さい", "ちいさい",
    "欲しい", "ほしい", "怖い", "こわい",
    "好き", "嫌い", "きらい", "大丈夫",
    "ずるい", "ふるい", "ぬるい", "えぐい",
    # all-sound-kana verbs, dictionary form
    "ふる", "ぬる", "ぬぐ", "つぐ", "ふく", "つく", "むく",
    "くぐる", "ふれる", "つれる", "くれる", "ぐれる",
    "ふるえる", "ふくれる", "くずれる", "つぶれる",
    "うごめく", "くずす", "ふくむ", "ふるう", "つぶす",
    # all-sound-kana nouns
    "ちち", "うち", "おく", "こえ", "ふじ", "おふろ", "ふくろ",
    "ちず", "こぶ", "こめ", "おこめ", "くず", "ふち", "つち",
    "いぬ", "ちえ", "ぬい", "えこ", "すじ", "うず",
    "こじ", "ふぐ", "めす", "つえ", "ふるえ",
    # all-sound-kana adverbs
    "すぐ", "ぐっ", "じっ",
    # all-sound-kana mimetics that do appear in dialogue
    "めちゃめちゃ", "ごちゃごちゃ", "ぐちゃぐちゃ", "ごろごろ", "ぐるぐる",
    "ぶつぶつ", "ぷちぷち", "じめじめ", "ぬめぬめ",
    "ぷるぷる", "ぶるぶる",
    # domain vocabulary, likewise all sound kana
    "おちんちん", "ちんちん", "おまんこ", "まんこ",
    "おっぱい", "乳首", "クリトリス",
    "精液", "中", "奥", "膣",
    "行く", "いく", "来る", "くる", "帰る", "かえる",
    "見る", "みる", "聞く", "きく", "言う", "いう",
    "思う", "おもう", "知る", "しる", "分かる", "わかる",
    "できる", "ある", "いる",
    "入れる", "いれる", "出す", "だす",
    "舐める", "なめる", "吸う", "すう",
    "触る", "さわる", "動く", "うごく",
    "やる", "する", "なる",
    "止める", "やめる", "やめて",
    "見て", "来て", "して", "出して", "入れて",
    "待って", "まって",
    "もっと", "ちょっと", "すごく", "とても",
    "まだ", "もう", "ずっと", "やっぱり",
    "そして", "でも", "けど", "だから",
    "本当", "ほんとう", "ほんと",
    "ダメ", "だめ", "駄目",
    "嫌", "イヤ",
    "うそ", "嘘",
    "バカ", "ばか", "馬鹿",
    "変態", "へんたい",
    "何", "なに",
]

_DIALOGUE_INTERJECTIONS = [
    "えっ", "あっ", "おっ", "わっ", "やっ",
    "おい", "ねえ", "ちょっと", "まあ",
    "さあ", "うん", "ううん", "いや", "へえ",
    "よし", "よっし", "やった", "しまった",
    "なるほど",
    # Added 2026-09-02 after the benchmark corpus showed the imported list
    # deleting these as pure sound: every one is spelled entirely from the
    # sound alphabet, so nothing but an explicit entry saves it.
    "じゃあ", "じゃ", "いっぱい", "おめえ", "いっちゃ", "ばいばい",
    "すっごい", "すっごく", "ぜんぜん", "やっぱ", "こっち", "そっち", "どっち",
]

_EVIDENCE_RE = re.compile(
    "|".join(
        re.escape(word)
        for word in sorted(
            set(_MULTI_CHAR_PARTICLES + _VERB_ENDINGS + _QUESTION_MARKERS
                + _COMMON_WORDS + _DIALOGUE_INTERJECTIONS),
            key=len,
            reverse=True,
        )
    )
)
_SINGLE_PARTICLE_RE = re.compile(f"[{_SINGLE_CHAR_PARTICLES}]")
_SENTENCE_END_PARTICLE_RE = re.compile(r"[よねさわ](?=$|[、。！？…\s])")

# は is both the topic particle and the spelling of a breath. It counts as
# evidence only when what follows it is not another breath sound.
_HA_SOUND_FOLLOWERS = frozenset("ぁあぃいぅうぇえぉおーっはひふへほ")


def _has_ha_particle(text: str) -> bool:
    for match in re.finditer("は", text):
        pos = match.end()
        if pos >= len(text):
            # Trailing は: a particle only if the character before it is not
            # itself part of a breath run (はぁは vs 私は).
            if pos >= 2:
                previous = text[pos - 2]
                if previous not in SOUND_KANA and previous not in PUNCTUATION_CHARS:
                    return True
            continue
        following = text[pos]
        if following in _HA_SOUND_FOLLOWERS or following in PUNCTUATION_CHARS:
            continue
        return True
    return False


# Emphatic gemination (すごい -> すっごい, ちがう -> ちっがう) multiplies the
# vocabulary without adding meaning. Dropping the small tsu and retrying costs
# one regex and can only ever keep more lines, never delete more.
_SMALL_TSU_RE = re.compile("[っッ]")


def has_language_evidence(text: str) -> bool:
    """Whether ``text`` shows any sign of being real dialogue."""
    if _KANJI_RE.search(text):
        return True
    if _EVIDENCE_RE.search(text):
        return True
    degeminated = _SMALL_TSU_RE.sub("", text)
    if degeminated != text and _EVIDENCE_RE.search(degeminated):
        return True
    if _has_ha_particle(text):
        return True
    if _SINGLE_PARTICLE_RE.search(text):
        return True
    if re.search(r"[?？]", text):
        return True
    if _SENTENCE_END_PARTICLE_RE.search(text):
        return True
    # Two or more distinct katakana are a loanword, not a vocalisation run.
    katakana = {ch for ch in text if "ァ" <= ch <= "ヶ"}
    return len(katakana) >= 2


def is_sound_only_line(text: str) -> bool:
    """Whether one line is nothing but sound-effect kana."""
    if not text or not _JAPANESE_RE.search(text):
        return False
    if has_language_evidence(text):
        return False
    for ch in text:
        if ch in PUNCTUATION_CHARS or ch in SOUND_KANA:
            continue
        return False
    return True


def is_nonverbal(text: str) -> bool:
    """Whether every Japanese line of ``text`` is pure vocalisation.

    Multi-line entries are judged line by line and kept whole if any line is
    dialogue -- WhisperJAV shipped the opposite (it returned on the first
    Japanese line) and deleted ``はぁはぁ\\nもう我慢できない`` entirely.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    japanese = [line for line in lines if _JAPANESE_RE.search(line)]
    if not japanese:
        return False
    return all(is_sound_only_line(line) for line in japanese)


# --- lone-token artifacts ---------------------------------------------------

# A whole line that is one of these, optionally with a single "。", is a moan
# onset or a truncated decode rather than speech. Deliberately narrow: multi
# token stutters (あ、あ。), laughter (はは。), long vowels (あー。) and
# backchannel (うん。 はい。) are a different problem and stay.
NONVERBAL_TOKENS = ("つ", "ふ", "ふっ", "切", "は", "え", "あ", "ん")

_LONE_TOKEN_RE = re.compile(
    "^(?:"
    + "|".join(sorted((re.escape(t) for t in NONVERBAL_TOKENS), key=len, reverse=True))
    + ")。?$"
)


def is_lone_nonverbal_token(text: str) -> bool:
    """Whether the whole line is a single curated non-verbal token."""
    if not text:
        return False
    return bool(_LONE_TOKEN_RE.match(text.strip()))


def removal_reason(text: str) -> str | None:
    """``"nonverbal"`` / ``"lone-token"`` if ``text`` should go, else ``None``."""
    if is_lone_nonverbal_token(text):
        return "lone-token"
    if is_nonverbal(text):
        return "nonverbal"
    return None

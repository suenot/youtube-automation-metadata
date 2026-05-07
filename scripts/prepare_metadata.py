#!/usr/bin/env python3
"""Prepare YouTube metadata from finished video + blog article.

Pipeline:
  1. Extract audio from video (ffmpeg)
  2. Transcribe with Whisper
  3. Generate SRT
  4. Research YouTube tags
  5. Generate EN + RU metadata (JSON + TXT)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter

import whisper

# ── Paths ──────────────────────────────────────────────────────────────────
BLOG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "marketmaker-cc-landing", "src", "content", "blog",
)
BLOG_DIR = os.path.abspath(BLOG_DIR)

TELEGRAM = "https://t.me/marketmaker_cc"
SITE = "https://marketmaker.cc"

# ── Tag research constants ─────────────────────────────────────────────────
BAD_PATTERNS = re.compile(
    r'\b(pdf|python|course|tutorial|app|audiobook|india|kya|hindi|'
    r'vevo|official|music video|clip|download|free|review|book|'
    r'salary|software|platform|bot|playlist|episode|part|chapter|'
    r' explained|for beginners|introduction|guide|using|vs)\b',
    re.IGNORECASE,
)
TOO_BROAD = {
    'trading', 'analysis', 'optimization', 'strategy', 'finance', 'investing',
    'stock', 'market', 'forex', 'crypto', 'bitcoin', 'money', 'profit',
    'data', 'research', 'code', 'test', 'backtest', 'overfit', 'optuna',
}
BRAND_TAGS = ['marketmaker', 'marketmaker_cc']

EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "in", "for", "on", "with", "at", "by", "from",
    "as", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "what", "which", "who", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now",
}
RU_STOPWORDS = {
    "и", "в", "не", "на", "я", "быть", "он", "с", "что", "а", "по", "это",
    "она", "к", "но", "мы", "как", "из", "у", "то", "за", "свой", "его",
    "весь", "вы", "для", "о", "же", "ну", "бы", "чтобы", "который", "от",
    "так", "этот", "тот", "такой", "все", "да", "нет", "или", "если",
    "тогда", "когда", "уже", "еще", "только", "даже", "вот", "всем",
    "можно", "надо", "нужно",
}


# ── Article parsing ────────────────────────────────────────────────────────
def parse_frontmatter(path: str) -> dict:
    data = {"title": "", "description": "", "tags": [], "body": ""}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        data["body"] = body.strip()
        for line in front.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "tags":
                    val = val.strip("[]")
                    data["tags"] = [t.strip().strip('"').strip("'") for t in val.split(",") if t.strip()]
                else:
                    data[key] = val
    else:
        data["body"] = text.strip()
    return data


# ── Audio extraction ───────────────────────────────────────────────────────
def extract_audio(video_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_path]
    print(f"Extracting audio: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Audio saved to {output_path}")


# ── Whisper transcription ──────────────────────────────────────────────────
def transcribe(audio_path: str, model_size: str = "base", language: str = None) -> dict:
    print(f"Loading Whisper model {model_size}...")
    model = whisper.load_model(model_size)
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    return result


# ── SRT generation ─────────────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(subs: dict, output_path: str):
    segments = subs.get("segments", [])
    lines = []
    for idx, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{format_time(seg['start'])} --> {format_time(seg['end'])}")
        lines.append(text)
        lines.append("")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"SRT saved to {output_path}")


# ── Timestamps from transcription (no slides) ─────────────────────────────
def _is_meaningful(text: str, min_words: int = 3) -> bool:
    words = text.split()
    if len(words) < min_words:
        return False
    return sum(1 for c in text if c.isalpha()) / max(len(text), 1) >= 0.5


def _normalize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return set(text.split())


def _ngrams(words: list, n: int = 2) -> set:
    return set(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def _score_text(query: str, ref_words: set, ref_bigrams: set) -> float:
    words = query.lower()
    words = re.sub(r"[^\w\s]", " ", words).split()
    if not words:
        return 0.0
    wset = set(words)
    bg = _ngrams(words, 2)
    return len(wset & ref_words) + len(bg & ref_bigrams) * 2.0


def build_timestamps_from_article(subs: dict, frontmatter: dict, max_chapters: int = 12) -> list:
    """Build chapter timestamps by matching article section headings to Whisper segments.

    Uses the same scoring + advance logic as video_maker's sync_slides.py:
    iterate segments, score against upcoming headings, advance when score
    clearly exceeds current heading.
    """
    segments = subs.get("segments", [])
    if not segments:
        return []

    # Extract section headings from article body
    body = frontmatter.get("body", "")
    headings = re.findall(r'^##+\s+(.+)', body, re.MULTILINE)
    if not headings:
        return []

    # Clean headings
    clean_headings = []
    for h in headings:
        h = h.strip()
        h = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', h)
        h = re.sub(r'[`*#]', '', h).strip()
        h = re.sub(r'^[\d.]+\s*', '', h)
        if len(h) < 3 or len(h) > 100:
            continue
        clean_headings.append(h)

    if not clean_headings:
        return []

    # Build section text blocks for each heading (heading + paragraph after it)
    section_blocks = []
    for heading in clean_headings:
        heading_idx = body.find(heading)
        if heading_idx == -1:
            section_blocks.append(heading)
            continue
        next_h = body.find("\n##", heading_idx + len(heading))
        section = body[heading_idx:next_h] if next_h != -1 else body[heading_idx:heading_idx + 500]
        section_blocks.append(section)

    # Pre-compute word sets and bigrams for each section
    section_words = [_normalize(block) for block in section_blocks]
    section_bigrams = [_ngrams(list(_normalize(block))) for block in section_blocks]

    # Walk through segments like sync_slides.py
    current_heading = 0
    heading_start_time = 0.0
    min_duration = 15.0
    advance_ratio = 1.3
    look_ahead = 3

    timeline = []

    for seg in segments:
        t0 = seg["start"]
        txt = seg.get("text", "")

        # Compute scores for current and upcoming headings
        scores = {}
        for idx in range(current_heading, min(len(clean_headings), current_heading + look_ahead + 1)):
            scores[idx] = _score_text(txt, section_words[idx], section_bigrams[idx])

        best_idx = current_heading
        best_score = scores.get(current_heading, 0.0)

        time_on_current = t0 - heading_start_time
        if time_on_current >= min_duration:
            for idx in range(current_heading + 1, min(len(clean_headings), current_heading + look_ahead + 1)):
                sc = scores.get(idx, 0.0)
                if sc > best_score * advance_ratio and sc > best_score + 1.0:
                    best_idx = idx
                    best_score = sc

        if best_idx != current_heading:
            timeline.append({"start": heading_start_time, "heading": current_heading})
            current_heading = best_idx
            heading_start_time = t0

    # Close final heading
    if segments:
        timeline.append({"start": heading_start_time, "heading": current_heading})

    # Build stamps from timeline
    stamps = []
    for entry in timeline:
        t = entry["start"]
        heading = clean_headings[entry["heading"]]
        m = int(t) // 60
        s = int(t) % 60
        stamps.append({"time": f"{m}:{s:02d}", "text": heading})

    return stamps[:max_chapters]


# ── Tag research ───────────────────────────────────────────────────────────
def get_youtube_suggestions(query: str) -> list:
    try:
        url = f"https://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if len(data) > 1 and isinstance(data[1], list):
            return [s for s in data[1] if isinstance(s, str)]
    except Exception as e:
        print(f"  suggest error for '{query}': {e}", file=sys.stderr)
    return []


def search_youtube_competitors(query: str, max_results: int = 10) -> list:
    try:
        cmd = ["yt-dlp", f"ytsearch{max_results}:{query}", "--skip-download", "--print", "%(title)s", "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except Exception as e:
        print(f"  yt-dlp error: {e}", file=sys.stderr)
    return []


def extract_phrases_from_titles(titles: list) -> list:
    phrases = []
    for title in titles:
        title = BAD_PATTERNS.sub(' ', title.lower())
        words = re.findall(r"[\w#-]+", title)
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                joined = ' '.join(gram)
                if len(joined) < 10:
                    continue
                if all(w in TOO_BROAD for w in gram):
                    continue
                phrases.append(joined)
    return phrases


def clean_tag(tag: str) -> str:
    tag = tag.strip().lower()
    tag = re.sub(r"[^\w\s#-]", "", tag)
    tag = re.sub(r"\b20\d{2}\b", "", tag)
    return re.sub(r"\s+", " ", tag).strip()


def is_good_tag(tag: str) -> bool:
    if not tag or len(tag) < 10 or len(tag) > 50 or ' ' not in tag:
        return False
    if BAD_PATTERNS.search(tag):
        return False
    if all(w in TOO_BROAD for w in tag.split()):
        return False
    return True


def extract_tags_from_article(frontmatter: dict, lang: str = "en", max_tags: int = 15) -> list:
    """Generate YouTube tags from article content: headings, key phrases, frontmatter tags."""
    tags = []

    # 1. Section headings → split into separate tags at separators
    body = frontmatter.get("body", "")
    headings = re.findall(r'^##+\s+(.+)', body, re.MULTILINE)
    for h in headings:
        h = h.strip()
        h = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', h)
        h = re.sub(r'[`*#]', '', h).strip()
        h = re.sub(r'^[\d.]+\s*', '', h)
        h = re.sub(r'^(The |A |An |Как |Почему |Что )', '', h, flags=re.IGNORECASE)
        if not h or len(h) < 3:
            continue
        # Split into parts at: (parentheses), em-dash, colon, comma
        parts = re.split(r'\s*[\(（]\s*', h)
        split_parts = []
        for p in parts:
            p = re.sub(r'[\)）]', '', p).strip()
            for sub in re.split(r'\s*[—–:;,]\s*', p):
                sub = sub.strip()
                if sub and len(sub) >= 3:
                    split_parts.append(sub)
        for tag_text in split_parts:
            words = tag_text.split()
            if len(words) >= 2 and len(tag_text) <= 50:
                tags.append(tag_text)

    # 2. Key phrases from description (2-3 word, meaningful)
    desc = frontmatter.get("description", "")
    desc_words = re.findall(r'[\wа-яА-ЯёЁ-]+', desc.lower())
    stopwords = RU_STOPWORDS if lang == "ru" else EN_STOPWORDS
    filtered = [w for w in desc_words if len(w) > 3 and w not in stopwords]
    for n in (3, 2):
        for i in range(len(filtered) - n + 1):
            phrase = " ".join(filtered[i:i+n])
            if 12 <= len(phrase) <= 45:
                if len(set(filtered[i:i+n])) > 1:
                    tags.append(phrase)

    # 3. Frontmatter tags + limited combos
    ftags = [t.strip() for t in frontmatter.get("tags", []) if len(t.strip()) > 3]
    tags.extend(ftags)
    # Only add combos that combine 2 different-concept tags
    for i in range(len(ftags)):
        for j in range(i + 1, len(ftags)):
            combo = f"{ftags[i]} {ftags[j]}"
            if 10 <= len(combo) <= 50:
                tags.append(combo)

    # Deduplicate, preserving order
    seen = set()
    unique = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            unique.append(t)

    # Add brand tags
    for bt in BRAND_TAGS:
        if bt not in [t.lower() for t in unique]:
            unique.append(bt)

    return unique[:max_tags]


# ── Keyword extraction ─────────────────────────────────────────────────────
def extract_keywords(text: str, lang: str = "en", top_n: int = 10) -> list:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    stopwords = RU_STOPWORDS if lang == "ru" else EN_STOPWORDS
    filtered = [w for w in words if len(w) > 3 and w not in stopwords]
    counts = Counter(filtered)
    return [w for w, _ in counts.most_common(top_n)]


# ── Description builder ────────────────────────────────────────────────────
def build_description(frontmatter: dict, stamps: list, tags: list, slug: str, lang: str) -> str:
    lines = []
    desc = frontmatter.get("description", "")
    if not desc:
        body = frontmatter.get("body", "")
        desc = body.split("\n\n")[0].strip("# ").strip() if body else ""
    lines.append(desc)

    lines.append("")
    lines.append("🔍 Timestamps:" if lang == "en" else "🔍 Таймкоды:")
    for s in stamps[:12]:
        lines.append(f"{s['time']} {s['text']}")

    lines.append("")
    lines.append("📖 Read the full article:" if lang == "en" else "📖 Полная статья:")
    lines.append(f"{SITE}/en/blog/post/{slug}")

    lines.append("")
    lines.append("💬 Discuss:" if lang == "en" else "💬 Обсудить:")
    lines.append(TELEGRAM)

    lines.append("")
    lines.append("🔗 Tags:")
    lines.append(", ".join(tags))

    lines.append("")
    if lang == "en":
        lines.append(f"👍 Subscribe to our Telegram channel {TELEGRAM} for more algorithmic trading content.")
    else:
        lines.append(f"👍 Подписывайтесь на телеграм канал {TELEGRAM}, чтобы получать больше материалов по алготрейдингу.")
    return "\n".join(lines)


# ── Problems builder ───────────────────────────────────────────────────────
def build_problems(stamps: list, video_title: str, lang: str = "en") -> str:
    lines = []
    q = video_title + ("" if video_title.endswith("?") else "?")
    lines.append(f"0:00 {q}")
    for s in stamps[:5]:
        lines.append(f"{s['time']} {s['text']}")
    return "\n".join(lines)


# ── Main pipeline ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--slug", required=True, help="Blog article slug")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--temp-dir", default="", help="Temp directory")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--skip-transcribe", action="store_true", help="Reuse existing subtitles.json")
    args = parser.parse_args()

    slug = args.slug
    output_dir = args.output_dir
    temp_dir = args.temp_dir or os.path.join(os.path.dirname(output_dir), "temp", slug)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # ── Step 1: Extract audio ──
    audio_path = os.path.join(temp_dir, "audio.m4a")
    if not os.path.exists(audio_path):
        extract_audio(args.video, audio_path)
    else:
        print(f"Audio already exists: {audio_path}")

    # ── Step 2: Transcribe ──
    subs_path = os.path.join(temp_dir, "subtitles.json")
    if args.skip_transcribe and os.path.exists(subs_path):
        print(f"Loading existing subtitles: {subs_path}")
        with open(subs_path, "r", encoding="utf-8") as f:
            subs = json.load(f)
    else:
        subs = transcribe(audio_path, model_size=args.whisper_model, language="en")
        with open(subs_path, "w", encoding="utf-8") as f:
            json.dump(subs, f, ensure_ascii=False, indent=2)
        print(f"Subtitles saved to {subs_path}")

    # ── Step 3: Generate SRT ──
    srt_path = os.path.join(output_dir, f"{slug}.srt")
    save_srt(subs, srt_path)

    # ── Step 4: Read blog articles ──
    en_article = parse_frontmatter(os.path.join(BLOG_DIR, f"{slug}.en.md"))
    ru_article = parse_frontmatter(os.path.join(BLOG_DIR, f"{slug}.ru.md"))

    # ── Step 5: Build timestamps from article headings matched to transcription ──
    stamps = build_timestamps_from_article(subs, en_article)

    # ── Step 6: Generate tags from article content (EN + RU) ──
    en_tags = extract_tags_from_article(en_article, lang="en")
    ru_tags = extract_tags_from_article(ru_article, lang="ru")
    print(f"EN tags ({len(en_tags)}): {', '.join(en_tags[:5])}...")
    print(f"RU tags ({len(ru_tags)}): {', '.join(ru_tags[:5])}...")

    # ── Step 7: Generate EN metadata ──
    en_title = en_article.get("title", slug.replace("-", " ").title())
    if len(en_title) > 90:
        en_title = en_title[:87] + "..."
    en_desc = build_description(en_article, stamps, en_tags, slug, lang="en")
    en_problems = build_problems(stamps, en_title, lang="en")

    en_metadata = {
        "title": en_title,
        "description": en_desc,
        "tags": en_tags,
        "timestamps": stamps,
        "category": "Education",
        "type": "Concept overview",
        "level": "Advanced",
        "problems": en_problems,
    }

    en_json_path = os.path.join(output_dir, f"{slug}_metadata.json")
    with open(en_json_path, "w", encoding="utf-8") as f:
        json.dump(en_metadata, f, ensure_ascii=False, indent=2)
    print(f"EN metadata JSON saved to {en_json_path}")

    en_txt_path = os.path.join(output_dir, f"{slug}_metadata.txt")
    with open(en_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title:\n{en_title}\n\n")
        f.write(f"Description:\n{en_desc}\n\n")
        f.write(f"Category:\nEducation\n\n")
        f.write(f"Type:\nConcept overview\n\n")
        f.write(f"Level:\nAdvanced\n\n")
        f.write(f"Problems:\n{en_problems}\n\n")
        f.write(f"Tags:\n{', '.join(en_tags)}\n")
    print(f"EN metadata TXT saved to {en_txt_path}")

    # ── Step 8: Generate RU metadata ──
    ru_title = ru_article.get("title", en_title)
    if len(ru_title) > 90:
        ru_title = ru_title[:87] + "..."
    ru_desc = build_description(ru_article, stamps, ru_tags, slug, lang="ru")
    ru_problems = build_problems(stamps, ru_title, lang="ru")

    ru_metadata = {
        "title": ru_title,
        "description": ru_desc,
        "tags": ru_tags,
        "timestamps": stamps,
        "category": "Education",
        "type": "Concept overview",
        "level": "Advanced",
        "problems": ru_problems,
    }

    ru_json_path = os.path.join(output_dir, f"{slug}_ru_metadata.json")
    with open(ru_json_path, "w", encoding="utf-8") as f:
        json.dump(ru_metadata, f, ensure_ascii=False, indent=2)
    print(f"RU metadata JSON saved to {ru_json_path}")

    ru_txt_path = os.path.join(output_dir, f"{slug}_ru_metadata.txt")
    with open(ru_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title:\n{ru_title}\n\n")
        f.write(f"Description:\n{ru_desc}\n\n")
        f.write(f"Category:\nEducation\n\n")
        f.write(f"Type:\nConcept overview\n\n")
        f.write(f"Level:\nAdvanced\n\n")
        f.write(f"Problems:\n{ru_problems}\n\n")
        f.write(f"Tags:\n{', '.join(ru_tags)}\n")
    print(f"RU metadata TXT saved to {ru_txt_path}")

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()

# Video YouTube Prepare

Automated pipeline for preparing YouTube metadata from finished videos and blog articles. Generates titles, descriptions, tags, timestamps, and SRT subtitles in both English and Russian.

## What It Does

Given a finished video file and a corresponding blog article slug, the pipeline:

1. **Extracts audio** from the video using FFmpeg
2. **Transcribes speech** to text using OpenAI Whisper
3. **Generates SRT subtitles** for YouTube upload
4. **Builds chapter timestamps** by matching article section headings to transcription segments
5. **Researches YouTube tags** from article content (headings, key phrases, frontmatter tags)
6. **Generates metadata** (title, description, tags, timestamps) in both EN and RU — as JSON and human-readable TXT

## Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) installed and available in `PATH`
- [OpenAI Whisper](https://github.com/openai/whisper) (`pip install openai-whisper`)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (optional, for tag research via YouTube search)

### Python Dependencies

```bash
pip install openai-whisper
```

## Project Structure

```
video_youtube_prepare/
├── scripts/
│   ├── prepare_metadata.py   # Main pipeline script
│   └── run.sh                # Batch runner for multiple videos
├── input/                    # Place source video files here
├── output/                   # Generated metadata, SRT files
└── temp/                     # Intermediate files (audio, subtitles JSON)
```

## Usage

### Single Video

```bash
python scripts/prepare_metadata.py \
    --video input/my_video.mp4 \
    --slug my-blog-article-slug \
    --output-dir output/my-blog-article-slug \
    --whisper-model base
```

### Batch Processing

Edit the `VIDEO_MAP` in `scripts/run.sh` to define video-to-slug mappings, then:

```bash
bash scripts/run.sh
```

### CLI Options

| Option | Description | Default |
|---|---|---|
| `--video` | Path to input video file | *required* |
| `--slug` | Blog article slug (used to find `.en.md` / `.ru.md` articles) | *required* |
| `--output-dir` | Directory for output files | *required* |
| `--temp-dir` | Directory for intermediate files | `temp/<slug>` |
| `--whisper-model` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) | `base` |
| `--skip-transcribe` | Reuse existing `subtitles.json` from temp dir | `false` |

## Output Files

For each video, the pipeline generates:

| File | Description |
|---|---|
| `<slug>.srt` | YouTube-ready SRT subtitles |
| `<slug>_metadata.json` | Structured EN metadata (title, description, tags, timestamps) |
| `<slug>_metadata.txt` | Human-readable EN metadata for copy-paste into YouTube Studio |
| `<slug>_ru_metadata.json` | Structured RU metadata |
| `<slug>_ru_metadata.txt` | Human-readable RU metadata |

## How It Works

### Timestamp Generation

Instead of fixed-interval timestamps, the tool matches blog article section headings against Whisper transcription segments using word overlap + bigram scoring. This produces accurate chapter markers that correspond to actual topic transitions in the video.

### Tag Generation

Tags are extracted directly from article content:
- Section headings split at separators (parentheses, colons, dashes)
- Key phrases from the article description (2-3 word meaningful combinations)
- Frontmatter tags and their pairwise combinations
- Brand tags (`marketmaker`, `marketmaker_cc`) are always appended

### Blog Article Integration

The tool expects blog articles in Markdown with YAML frontmatter at a configurable path (default: `../../marketmaker-cc-landing/src/content/blog/`). Articles should have `title`, `description`, and `tags` fields, plus section headings (`##`) in the body.

## Related Projects

- [video_maker](../video_maker/) — generates the video itself from audio + PDF slides

## License

MIT

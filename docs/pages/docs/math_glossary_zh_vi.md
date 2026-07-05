# Chinese -> Vietnamese Math Glossary

## Purpose

Use a master JSON glossary to improve Chinese math lecture transcription and translation.

## Flow

1. Upload/select glossary JSON in Settings.
2. Run b.1-3 Draft Translation.
3. After ASR and sentence segmentation, VideoLingo normalizes transcript with glossary.
4. VideoLingo builds custom_terms.xlsx.
5. Summarization reads custom_terms.xlsx and writes output/log/terminology.json.
6. Translation uses terminology.json in Points to Note.

## Why not inject the whole JSON?

Because it is too long, expensive, and noisy. We filter only terms relevant to the current transcript.

## CLI

```bash
python tools/build_custom_terms_from_glossary.py --normalize-source
```

## Files

- glossaries/olympiad_math_zh_vi_glossary.json
- custom_terms.xlsx
- output/log/split_by_meaning.txt
- output/log/split_by_meaning.before_glossary.txt
- output/log/terminology.json

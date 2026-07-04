import pandas as pd
from pathlib import Path

input_file = Path("output/log/cleaned_chunks.xlsx")
output_file = Path("output/recognized_from_cleaned_chunks.srt")

df = pd.read_excel(input_file)
df = df.dropna(subset=["text", "start", "end"]).copy()
df["text"] = df["text"].astype(str).str.strip().str.strip('"“” ')
df = df[df["text"] != ""]
df = df.sort_values(["start", "end"]).reset_index(drop=True)

MAX_CHARS = 28
MAX_DURATION = 5.0
GAP_SPLIT = 0.8
PUNCT_SPLIT = "。！？；!?;"

def fmt_time(sec):
    sec = max(0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    if s >= 60:
        m += 1
        s -= 60
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

entries = []
buf = []
start = None
last_end = None

for _, row in df.iterrows():
    text = str(row["text"]).strip()
    st = float(row["start"])
    en = float(row["end"])

    if start is None:
        start = st
        last_end = en
        buf = [text]
        continue

    current_text = "".join(buf)
    gap = st - last_end
    duration = last_end - start

    should_split = (
        gap > GAP_SPLIT
        or len(current_text) >= MAX_CHARS
        or duration >= MAX_DURATION
        or (current_text and current_text[-1] in PUNCT_SPLIT)
    )

    if should_split:
        final_text = "".join(buf).strip()
        if final_text:
            entries.append((start, last_end, final_text))
        start = st
        buf = [text]
    else:
        buf.append(text)

    last_end = en

if buf:
    final_text = "".join(buf).strip()
    if final_text:
        entries.append((start, last_end, final_text))

with open(output_file, "w", encoding="utf-8-sig", newline="\n") as f:
    for i, (st, en, text) in enumerate(entries, 1):
        if en <= st:
            en = st + 0.5
        f.write(f"{i}\n")
        f.write(f"{fmt_time(st)} --> {fmt_time(en)}\n")
        f.write(f"{text}\n\n")

print(f"Done: {output_file}")
print(f"Entries: {len(entries)}")
for i, item in enumerate(entries[:20], 1):
    print(i, fmt_time(item[0]), "-->", fmt_time(item[1]), item[2])

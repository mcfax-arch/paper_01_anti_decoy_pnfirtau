"""Prepare a clean Markdown copy for DOCX conversion (decode \\uXXXX, drop duplicate title/author)."""
import re

src = 'paper_en.md'
dst = 'submission/paper_en_clean.md'

text = open(src, encoding='utf-8').read()

# Decode literal JSON-style escapes such as \u2014 that leaked into the Markdown.
text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), text)

lines = text.splitlines()
out = []
skipped_title = False
for line in lines:
    if not skipped_title and line.startswith('# Attention with a Calibrated Off Switch'):
        skipped_title = True
        continue
    if line.startswith('**Author:** Maxim Zhukov'):
        continue
    if line.startswith('**Positioning.**') or line.startswith('**Repository & Reproducibility Package:**'):
        continue
    out.append(line)

open(dst, 'w', encoding='utf-8').write('\n'.join(out))
print('wrote', dst, 'chars', len('\n'.join(out)))
print('first lines:')
for line in out[:6]:
    print(repr(line[:100]))

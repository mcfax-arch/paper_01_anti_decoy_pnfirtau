"""Scrub internal host paths and round labels from results/*.json before public release.

Originals are backed up under .agency/release_backup_results/ (excluded from git).
Only metadata strings are changed; numeric values are untouched.
"""
import json
import re
import shutil
from pathlib import Path

root = Path('.')
results = root / 'results'
backup = root / '.agency' / 'release_backup_results'
backup.mkdir(parents=True, exist_ok=True)

path_re = re.compile(r'/mnt/i/FFT_Attention[^"\\]*')
replacements = [
    ('PAPER_01_REWORK_PLAN.md', 'README.md'),
    ('_backup_pre_c1', 'previous_reference'),
    ('C1 matched', 'matched-precision'),
    ('C2:', 'multi-seed:'),
    ('Close audit R2', 'Close audit'),
]

total = 0
files = 0
for p in sorted(results.glob('*.json')):
    text = p.read_text(encoding='utf-8')
    new = path_re.sub('<checkpoint_path>', text)
    for old, rep in replacements:
        new = new.replace(old, rep)
    if new != text:
        shutil.copy2(p, backup / p.name)
        p.write_text(new, encoding='utf-8')
        files += 1
        total += sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b)
    json.loads(p.read_text(encoding='utf-8'))  # validate JSON

print('scrubbed files', files, 'changed lines', total)
print('backup', backup)

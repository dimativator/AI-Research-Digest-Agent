"""User-requested fresh retry; preserve the original ten-attempt experiment."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from eval.run_weekly import SharedFixtures, run_agent, save

root = Path(__file__).resolve().parent
original = root.parents[1]
manifest = json.loads((original / 'manifest.json').read_text())
for name, digest in manifest['source_sha256'].items():
    assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest, name
for name in ['profile.json', 'corpus.json', 'rankings.json', 'e5-details.json']:
    shutil.copyfile(original / name, root / name)
task = root / 'retry-01'
task.mkdir(exist_ok=True)
for name in ['corpus.json', 'rankings.json', 'e5-details.json']:
    shutil.copyfile(root / name, task / name)
if not (root / 'fixtures').exists():
    shutil.copytree(original / 'fixtures', root / 'fixtures')
save(root / 'manifest.json', {**manifest, 'repetitions': 1, 'workers': 1,
    'retry_of': '../../repeat-18/debate/outcome.json',
    'protocol': 'One user-requested fresh retry. Same model, corpus, rankings, profile and budgets. Fresh screening and model calls, copied tool fixtures. Original failure remains in original experiment statistics.'})
run_agent(root, manifest, dict(task='retry-01', date_from=manifest['date_from'], date_to=manifest['date_to']), 'debate', SharedFixtures(root / 'fixtures'), digest_size=10)
result = json.loads((task / 'debate/outcome.json').read_text())
labels = {p['paper_id']: p['label'] for p in json.loads((original / 'sample.json').read_text())['papers']}
labels['2611.07859'] = 1
if result['status'] == 'completed':
    result['relevance_at_10'] = sum(labels[p] for p in result['selected']) / 10
    result['selected_labels'] = [labels[p] for p in result['selected']]
save(root / 'retry-summary.json', result)
print('RETRY_RESULT', json.dumps(result), flush=True)

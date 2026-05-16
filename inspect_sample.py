import json
with open('results/preprocessed_target.json') as f:
    data = json.load(f)
sample = next(s for s in data if s['sample_token'].startswith('17e2c5b3'))
print(json.dumps(sample, indent=2, default=str))

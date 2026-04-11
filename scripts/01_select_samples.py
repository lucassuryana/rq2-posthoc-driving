import json
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
with open('data/drivelm/v1_1_val_nus_q_only.json') as f:
    dlm = json.load(f)

mini_tokens = {s['token'] for s in nusc.sample}
selected = []

for scene_token, scene_data in dlm.items():
    for sample_token, entry in scene_data['key_frames'].items():
        if sample_token not in mini_tokens:
            continue
        qa = entry.get('QA', {})
        if not qa.get('planning'):
            continue
        sample = nusc.get('sample', sample_token)
        future_tokens = []
        tok = sample['next']
        for _ in range(3):
            if tok == '':
                break
            future_tokens.append(tok)
            tok = nusc.get('sample', tok)['next']
        if len(future_tokens) < 3:
            continue
        question = qa['planning'][0]['Q']
        selected.append({
            'sample_token': sample_token,
            'scene_token': scene_token,
            'question': question,
            'future_tokens': future_tokens[:3],
        })

print(f"Selected: {len(selected)} samples")
with open('results/selected_samples.json', 'w') as f:
    json.dump(selected, f, indent=2)
print("Saved to results/selected_samples.json")

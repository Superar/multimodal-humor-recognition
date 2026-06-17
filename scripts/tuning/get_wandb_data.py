from pathlib import Path

import pandas as pd
from tqdm import tqdm
import wandb


project = 'multimodal-humor-recognition'
entity = 'limma_'
savepath = Path('results/sweeps/')
sweeps_path = savepath / 'sweeps.parquet'
runs_path = savepath / 'runs.parquet'
history_path = savepath / 'history.parquet'

# ===== Load local data =====
print('Loading local data')
savepath.mkdir(exist_ok=True, parents=True)
local_runs = set()
local_runs_df = pd.DataFrame()
local_history_df = pd.DataFrame()
if runs_path.exists():
    local_runs_df = pd.read_parquet(runs_path)
    local_runs = set(local_runs_df['run name'])
if history_path.exists():
    local_history_df = pd.read_parquet(history_path)
print(f'Found {len(local_runs)} local runs saved')

# ====== Fetch Sweeps =====
api = wandb.Api()
sweeps = api.project(project, entity).sweeps()
params = ['batch_size', 'epochs', 'freeze_base', 'hidden_dim', 'learning_rate']

sweeps_data = {'sweep id': [], 'model': [], 'method': []}
for sweep in sweeps:
    if sweep.state == 'FINISHED':
        model, method = sweep.name.split(' + ')
        sweeps_data['sweep id'].append(sweep.id)
        sweeps_data['model'].append(model)
        sweeps_data['method'].append(method)
sweeps_df = pd.DataFrame.from_dict(sweeps_data)
sweeps_ids = sweeps_df['sweep id'].unique().tolist()
print(f'Fetched {len(sweeps_ids)} sweeps')

# ===== Fetch Runs =====
print('Fetching runs')
runs = api.runs(
    f'{entity}/{project}',
    filters={'sweep': {'$in': sweeps_ids}, 'state': 'finished'},
)
total_runs = len(runs)

new_runs = []
new_histories = []
for run in tqdm(runs, total=total_runs, desc='Fetching W&B runs'):
    if run.name in local_runs:
        continue
    run_config = {param: run.config[param] for param in params}
    run_config['run name'] = run.name
    run_config['sweep id'] = run.sweep.id
    new_runs.append(run_config)

    run_history = pd.DataFrame(
        run.scan_history(
            keys=['step', 'train/loss', 'eval/f1_macro', 'eval/accuracy']
        )
    )
    run_history['run name'] = run.name
    new_histories.append(run_history)

if new_runs:
    print(f'Fetched {len(new_runs)} new runs')

    new_runs_df = pd.DataFrame(new_runs)
    new_history_df = pd.concat(new_histories, ignore_index=True)

    runs_df = pd.concat([local_runs_df, new_runs_df], ignore_index=True)
    history_df = pd.concat([local_history_df, new_history_df], ignore_index=True)

    runs_df.to_parquet(runs_path, index=False)
    history_df.to_parquet(history_path, index=False)
sweeps_df.to_parquet(sweeps_path, index=False)

# Prometheus source adapter

This directory exposes StarVLA's native training and model-server entrypoints
to Prometheus without importing the model stack during validation. It is a
source-side capability declaration, not a source pin; the consuming
Prometheus `policy/star_vla` branch records the exact gitlink revision.

Inspect the repository contract:

```bash
python .prometheus/adapter.py capabilities
python .prometheus/adapter.py doctor
```

Plan native StarVLA training without executing it:

```bash
python .prometheus/adapter.py train --plan -- \
  --config_yaml /managed/configs/starvla.yaml \
  --run_root_dir /managed/runs \
  --run_id example
```

Accelerate launcher tokens are passed separately, one token per option:

```bash
python .prometheus/adapter.py train --plan \
  --accelerate-arg=--num_machines --accelerate-arg=2 \
  --accelerate-arg=--machine_rank --accelerate-arg=0 \
  -- --config_yaml /managed/configs/starvla.yaml
```

`resume` requires `--checkpoint` and deliberately sets
`trainer.is_resume=false`: StarVLA's committed checkpoint path restores model
weights but does not save/restore optimizer state. This is `weights_only`, not
full-state resume.

There is no universal dataset preparation, evaluation, or export command in
StarVLA. Those stages remain unsupported here. `serve` dispatches only the
native model server and does not authorize hardware rollout.

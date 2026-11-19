#!/bin/bash
# After all training runs: per-image confusion (both decodings), complexity, application to the KartaView photos (headline VCH-SegFormer-B5 and SegFormer-B5, and the MiT-B2 pair), example predictions.
: "${WORK:?set WORK to the working directory of the GPU jobs (data/vegstruct, runs/, logs/)}"; : "${REPO:?set REPO to the clone of this repository}"
W=$WORK; cd $W; export PYTHONPATH=$REPO/code/seg
export HF_HOME=$W/cache/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp
PY=${PY:-python}; G=${1:-1}
$PY $REPO/code/seg/eval_perimage.py --data data/vegstruct --runs runs --gpu $G > logs/post_perimage.log 2>&1
$PY $REPO/code/seg/complexity.py --out runs/complexity.csv --gpu $G > logs/post_complexity.log 2>&1
for v in vch_b5 base_b5 vch base; do $PY $REPO/code/seg/apply_kv.py --ckpt runs/$v/best.pt --variant $v --img data/kv/img512 --out runs/$v/kv_shares.csv --gpu $G > logs/post_apply_$v.log 2>&1; done
$PY $REPO/code/seg/predict_examples.py --data data/vegstruct --runs runs --kv data/kv/img512 --out runs/examples --gpu $G > logs/post_examples.log 2>&1
echo DONE >> logs/post_runs.done

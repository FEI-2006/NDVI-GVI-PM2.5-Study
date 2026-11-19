#!/bin/bash
# usage: run_queue.sh GPU ITEM [ITEM ...]; ITEM = variant (seed 20260929, runs/<variant>) or variant:seed (runs/<variant>_s<seed>);
# variants: base vcg hvh vch (MiT-B2; base_b5 vch_b5 ... for the MiT-B5 encoder), unet pspnet deeplab, upcnxt upswint (UperNet with ConvNeXt-T / Swin-T encoder).
# A finished run (test_metrics.json) is skipped and an interrupted one resumes from last.pt, so the same command can be re-issued after a reboot.
: "${WORK:?set WORK to the working directory of the GPU jobs (data/vegstruct, runs/, logs/)}"; : "${REPO:?set REPO to the clone of this repository}"
W=$WORK; cd $W; export PYTHONPATH=$REPO/code/seg
export HF_HOME=$W/cache/hf HF_HUB_OFFLINE=1 TMPDIR=$W/tmp
PY=${PY:-python}; G=$1; shift
for item in "$@"; do
  v=${item%%:*}; s=${item#*:}
  if [ "$s" = "$item" ]; then s=20260929; out=runs/$v; else out=runs/${v}_s$s; fi
  [ -f $out/test_metrics.json ] && continue
  $PY $REPO/code/seg/train_seg.py --variant $v --seed $s --data data/vegstruct --out $out --gpu $G --workers 6 >> logs/train_$(basename $out).log 2>&1
done

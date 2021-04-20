cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --gpu $gpu \
    --aggr-gpu $gpu \
    --seed 0 \
    --dataset ppi \
    --inductive \
    --threshold 0.9 \
    --epoch-setting 2000 2000 2000\
    --eval-every 10 \
    --lr 0.001 \
    --batch-size 256 \
    --mlp-layer 3 \
    --num-hidden 1024 \
    --dropout 0.3 \
    --attn-drop 0.1 \
    --input-drop 0.0 \
    --K 2 \
    --label-K 9 \
    --use-labels \
    --weight-decay 3e-6 \
    --warmup-stage -1 \
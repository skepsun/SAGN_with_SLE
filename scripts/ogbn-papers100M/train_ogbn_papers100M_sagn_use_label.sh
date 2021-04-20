cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset ogbn-papers100M \
    --gpu $gpu \
    --aggr-gpu -1 \
    --eval-every 1 \
    --model sagn \
    --seed 0 \
    --num-runs 10 \
    --threshold 0.7 \
    --epoch-setting 100 50 50 \
    --lr 0.001 \
    --batch-size 5000 \
    --num-hidden 1024 \
    --dropout 0.4 \
    --attn-drop 0.0 \
    --input-drop 0.0 \
    --K 3 \
    --label-K 9 \
    --use-labels \
    --weight-decay 0 \
    --warmup-stage -1
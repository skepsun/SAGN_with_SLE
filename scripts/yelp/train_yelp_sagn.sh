   cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu" 
    --seed 0 \
    --num-runs 10 \
    --model sagn \
    --aggr-gpu $gpu \
    --gpu $gpu \
    --dataset yelp \
    --inductive \
    --eval-every 1 \
    --threshold 0.9 \
    --epoch-setting 100 100 100\
    --lr 0.0001 \
    --batch-size 200 \
    --eval-batch-size 200000 \
    --mlp-layer 2 \
    --num-hidden 256 \
    --dropout 0.05 \
    --attn-drop 0.0 \
    --input-drop 0.0 \
    --K 2 \
    --label-K 6 \
    --weight-decay 5e-6 \
    --warmup-stage -1
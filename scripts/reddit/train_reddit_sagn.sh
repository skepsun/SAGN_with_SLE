cd "$(dirname $0)" 
if [ ! -n "$1" ] ; then
    gpu="0"
else
    gpu="$1"
fi
echo "gpu: $gpu"
python -u ../../src/sagn.py \
    --dataset reddit \
    --gpu $gpu \
    --aggr-gpu $gpu \
    --model sagn \
    --inductive \
    --threshold 0.9 \
    --epoch-setting 500 500 500 \
    --lr 0.0001 \
    --batch-size 1000 \
    --num-hidden 512 \
    --dropout 0.7 \
    --attn-drop 0.0 \
    --input-drop 0.0 \
    --K 2 \
    --use-labels \
    --label-K 4 \
    --weight-decay 0 \
    --warmup-stage -1
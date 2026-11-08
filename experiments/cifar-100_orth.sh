# bash experiments/cifar-100_orth.sh
# Same as experiments/cifar-100.sh (same config, seeds, epochs, prompt parameters).
# The only change: a 6th --prompt_param value = strength of the orthogonality regularization
# on the prefix keys (0 = original SMoPE). One training run per strength.
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 sh experiments/cifar-100_orth.sh for OOM error prevention
DATASET=cifar-100
OUTDIR=task2/${DATASET}/10-task

GPUID='0'
REPEAT=2
OVERWRITE=1

mkdir -p $OUTDIR

for MU in 0.1 0.001 1 10 #try 0.001 0.01 0.1 1 10
do
python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 $MU --seeds 0 1 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --log_dir ${OUTDIR}/one-prompt-orth-${MU} \
    --batch_size 32
done

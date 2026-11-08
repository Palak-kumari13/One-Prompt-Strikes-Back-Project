# bash experiments/cifar-100_ec.sh
# Same as experiments/cifar-100.sh (same config, seeds, epochs, prompt parameters).
# The only differences: expert-choice routing is switched on, and the results go to a different folder.
DATASET=cifar-100
OUTDIR=task1/${DATASET}/10-task

GPUID='0'
REPEAT=2
OVERWRITE=1

mkdir -p $OUTDIR

# SMOPE_ROUTING=expert_choice -> every expert picks its best inputs from the minibatch (models/vit.py)
# SMOPE_CAPACITY=1.0          -> capacity = round(B*K/E); raise/lower it to give experts more/fewer inputs
SMOPE_ROUTING=expert_choice SMOPE_CAPACITY=1.0 \
python -u run.py --config configs/cifar-100_prompt_smope.yaml --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 1e-5 0.4 --seeds 0 1 \
    --crct_epochs 50 --ca_batch_size_ratio 1 \
    --batch_size 32 \
    --log_dir ${OUTDIR}/one-prompt-ec
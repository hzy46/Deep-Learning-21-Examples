rm -rf /tmp/nmt_model_zh
mkdir -p /tmp/nmt_model_zh
python -m nmt.nmt \
--src=en --tgt=zh \
--attention=scaled_luong \
--vocab_prefix=/tmp/nmt_zh/vocab \
--train_prefix=/tmp/nmt_zh/train \
--dev_prefix=/tmp/nmt_zh/dev \
--test_prefix=/tmp/nmt_zh/test \
--out_dir=/tmp/nmt_model_zh \
--step_per_stats 100 \
--num_train_steps 300000 \
--num_layers 3 \
--num_units 512 \
--dropout 0.2 \
--metrics bleu;


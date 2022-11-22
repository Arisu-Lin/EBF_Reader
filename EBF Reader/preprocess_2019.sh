INPUT_TRAIN_FILE="data_2019/train.json"
INPUT_DEV_FILE="data_2019/dev.json"
INPUT_TEST_FILE="data_2019/test.json"

OUTPUT_DIR="data" #this dir must the same as the data_dir in train.sh

# mkdir ${OUTPUT_DIR}
tokenizer_path='../roberta-base-chinese'
# https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
python data_process.py \
   --tokenizer_path=$tokenizer_path \
   --full_data=${INPUT_TRAIN_FILE} \
   --example_output=${OUTPUT_DIR}/train_example.pkl.gz \
   --feature_output=${OUTPUT_DIR}/train_feature.pkl.gz \

python data_process.py \
   --tokenizer_path=$tokenizer_path \
   --full_data=${INPUT_DEV_FILE} \
   --example_output=${OUTPUT_DIR}/dev_example.pkl.gz \
   --feature_output=${OUTPUT_DIR}/dev_feature.pkl.gz \

python data_process.py \
    --tokenizer_path=$tokenizer_path \
    --full_data=${INPUT_TEST_FILE} \
    --example_output=${OUTPUT_DIR}/test_example.pkl.gz \
    --feature_output=${OUTPUT_DIR}/test_feature.pkl.gz \

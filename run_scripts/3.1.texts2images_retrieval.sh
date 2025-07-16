DATAPATH="../Chinese-CLIP-DATA"
dataset_name="Flickr30k-CN"
split=test # 指定计算valid或test集特征

python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
cat output.json
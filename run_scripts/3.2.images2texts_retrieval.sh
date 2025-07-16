DATAPATH="../Chinese-CLIP-DATA"
dataset_name="Flickr30k-CN"
split=test # 指定计算valid或test集特征

# For image-to-text retrieval, run the commands first to
# transform text-to-image jsonls to image-to-text ones:
python cn_clip/eval/transform_ir_annotation_to_tr.py \
        --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl

# After that, run the following commands
python cn_clip/eval/evaluation_tr.py \
        ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
        ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
        output.json
cat output.json
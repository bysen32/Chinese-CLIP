
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
DATAPATH="/workspace/Chinese-CLIP-DATA"
dataset_name="Flickr30k-CN"

# resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-h-14.pt

split=test # 指定计算valid或test集特征
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-base-chinese

split=valid # 指定计算valid或test集特征
python -u cn_clip/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
    --resume=${resume} \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-base-chinese

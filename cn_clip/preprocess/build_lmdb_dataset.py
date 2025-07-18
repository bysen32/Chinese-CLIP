# -*- coding: utf-8 -*-
'''
This script serializes images and image-text pair annotations into LMDB files,
which supports more convenient dataset loading and random access to samples during training 
compared with TSV and Jsonl data files.
'''

import argparse
import os
from tqdm import tqdm
import lmdb
import json
import pickle


'''
${split}_texts.jsonl
文本信息及图文对匹配关系保存在 ${split}_texts.jsonl 文件中，格式如下：
{
    'text_id': 'text_id_1',
    'text': 'This is a text annotation.',
    'image_ids': ['image_id_1', 'image_id_2']
}

${split}_imgs.tsv
为提升文件处理效率，将图片以base64形式分别存放在 ${split}_imgs.tsv 文件中。
每行表示一张图片，包含图片id 与 图片 base64，以tab隔开
1000002	/9j/4AAQSkZJ...YQj7314oA//2Q==

最后将 tsv 和 jsonl 文件一起序列化，
转换为内存索引的LMDB数据库文件，方便训练时随机读取
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
            (e.g. train,valid,test)"
    )
    parser.add_argument(
        "--lmdb_dir", type=str, default=None, help="specify the directory which stores the output lmdb files. \
            If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert os.path.isdir(args.data_dir), "The data_dir does not exist! Please check the input args..."

    # read specified dataset splits
    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))

    # build LMDB data files
    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir, "lmdb")

    for split in specified_splits:
        # open new LMDB files
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrite an existing LMDB file {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)

        lmdb_img = os.path.join(lmdb_split_dir, "imgs")
        env_img = lmdb.open(lmdb_img, map_size=1024**4)
        txn_img = env_img.begin(write=True)

        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024**4)
        txn_pairs = env_pairs.begin(write=True)

        # write LMDB file storing (image_id, text_id, text) pairs
        pairs_annotation_path = os.path.join(args.data_dir, "{}_texts.jsonl".format(split))
        with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
            write_idx = 0
            for line in tqdm(fin_pairs):
                line = line.strip()
                obj = json.loads(line)
                # Check valid field
                for field in ("text_id", "text", "image_ids"):
                    assert field in obj, "Field {} does not exist in line {}. \
                        Please check the integrity of the text annotation Jsonl file."

                for image_id in obj["image_ids"]:
                    dump = pickle.dumps((image_id, obj['text_id'], obj['text'])) # encoded (image_id, text_id, text)
                    txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)  
                    write_idx += 1
                    if write_idx % 5000 == 0:
                        txn_pairs.commit()
                        txn_pairs = env_pairs.begin(write=True)
            
            txn_pairs.put(key=b'num_samples',
                          value="{}".format(write_idx).encode('utf-8'))
            txn_pairs.commit()
            env_pairs.close()
        print("Finished serializing {} {} split pairs into {}.".format(write_idx, split, lmdb_pairs))

        # write LMDB file storing image base64 strings
        base64_path = os.path.join(args.data_dir, "{}_imgs.tsv".format(split))
        with open(base64_path, "r", encoding="utf-8") as fin_imgs:
            write_idx = 0
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)
            txn_img.put(key=b'num_images',
                        value="{}".format(write_idx).encode('utf-8'))
            txn_img.commit()
            env_img.close()                
        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img))

    print("done!")

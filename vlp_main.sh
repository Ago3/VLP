python vlp/decode_img2txt.py \
    --model_recover_path /media/SHARED/HDD1_2TB/vlp_checkpoint/coco_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.28.bin \
    --new_segment_ids --batch_size 10 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/vlp_data/COCO/region_feat_gvd_wo_bgd/ --split test \
    --src_file /media/SHARED/HDD1_2TB/vlp_data/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/vlp_data/COCO/annotations/coco_valid_jpgs.json

python main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/vlp_checkpoint/coco_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.28.bin \
    --new_segment_ids --batch_size 10 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/vlp_data/COCO/region_feat_gvd_wo_bgd/ --split test \
    --src_file /media/SHARED/HDD1_2TB/vlp_data/COCO/annotations/dataset_coco.json \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/vlp_data/COCO/annotations/coco_valid_jpgs.json

python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 10 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/vlp_data/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic \
    --region_bbox_file bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json

python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 1 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json

#Missing instances
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 1 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/missing_dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/all_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json


#VQA inference
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/vqa2_g2_lr2e-5_batch512_ft_from_s0.75_b0.25/model.19.bin \
    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/15_val.tsv.tmp_vqa_question.npy --batch_size 25 \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/all_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test

CUDA_VISIBLE_DEVICES="" python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 1 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/missing_dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/new_missing_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json

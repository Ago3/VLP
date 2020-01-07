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
CUDA_LAUNCH_BLOCKING=1 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 1 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/missing_dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/new_missing_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json

#VQA inference
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/new/cc/model.1.bin \
    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/babelpic_gold_vqa_sensemb.npy --batch_size 25 \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/all_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test

#VQA inference on imagenet
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/new/model.6.bin \
    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/imagenet/ \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/imagenet_dataset.tsv_vqa_question.npy --batch_size 25 \
    --region_det_file_prefix feat_cls_1000/imagenet/ \
    --region_bbox_file feat_cls_1000/imagenet/bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test

#VQA inference on silver
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/new/model.6.bin \
    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/silver/ \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/glosses_silver_batch_0_filtered_second_half.tsv_vqa_question.npy --batch_size 25 \
    --region_det_file_prefix feat_cls_1000/silver/ \
    --region_bbox_file feat_cls_1000/silver/bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test

#VQA for sensembedding
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/babelpic_gold_vqa_sensemb.npy --batch_size 25 \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/all_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test --output_dir /media/SHARED/HDD1_2TB/acalabrese/vlp_data/output --sensemb

#Train VQA
python3 main.py --output_dir /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/new/cc \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/new/cc/model.0.bin \
    --do_train --train_batch_size 10 --learning_rate 2e-5 --new_segment_ids --always_truncate_tail --amp \
    --num_train_epochs 20 --enable_butd --s2s_prob 0 --bi_prob 1 \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ \
    --tasks vqa2 --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/15_train.tsv.tmp_vqa_question.npy \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/all_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json \
    --mask_prob 0 --max_pred 1


CUDA_VISIBLE_DEVICES="" python3 main.py \
    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
    --new_segment_ids --batch_size 1 --beam_size 5 --enable_butd \
    --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/babelpic/ --split test \
    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/missing_dataset_babelpic.json \
    --region_det_file_prefix feat_cls_1000/babelpic/ \
    --region_bbox_file feat_cls_1000/babelpic/new_missing_bboxes.pkl \
    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json

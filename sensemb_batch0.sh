for half in "f" "s"
do
	CUDA_VISIBLE_DEVICES=0 python3 main.py \
	    --model_recover_path /media/SHARED/HDD1_2TB/acalabrese/vlp_checkpoint/cc_g8_lr1e-4_batch512_s0.75_b0.25/model.30.bin \
	    --new_segment_ids --enable_butd --image_root /media/SHARED/HDD1_2TB/acalabrese/detectron_data/silver/ \
	    --src_file /media/SHARED/HDD1_2TB/acalabrese/vlp_data/silver_0_first_half_vqa_sensemb.npy --batch_size 25 \
	    --region_det_file_prefix feat_cls_1000/silver/ \
	    --region_bbox_file feat_cls_1000/silver/batch_0_second_half_bboxes.pkl \
	    --file_valid_jpgs /media/SHARED/HDD1_2TB/acalabrese/vlp_data/COCO/annotations/coco_valid_jpgs.json --split test --output_dir /media/SHARED/HDD1_2TB/acalabrese/vlp_data/output/silver/0_${half}h --sensemb
done
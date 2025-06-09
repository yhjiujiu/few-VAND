### train on the MVTec AD dataset
CUDA_VISIBLE_DEVICES=1 nohup python train_few_score_map.py --dataset mvtec --train_data_path ./data/mvtec \
--data_path ./data/mvtec --save_path ./exps/visa/vit_large_14_518_score_map --config_path ./open_clip/model_configs/ViT-L-14-336.json --model ViT-L-14-336 \
--k_shot 4 --features_list 24 --pretrained openai --image_size 518  --batch_size 8 --aug_rate 0.2 --print_freq 1 \
--epoch 10 --save_freq 1 >./logs/MVTec_few_map_gpu1.txt 2>&1 &



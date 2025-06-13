### test on the VisA dataset
CUDA_VISIBLE_DEVICES=1 nohup python test_few.py --mode few_shot --dataset visa \
--data_path ./data/visa --save_path ./results/visa/few_shot/4shot/seed42_1 \
--config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/visa/vit_large_14_518/epoch_1.pth \
--model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
--pretrained openai --image_size 518 --k_shot 4 --seed 42 >./logs/visa_test_few_2class_1.txt 2>&1 &

### test on the MVTec AD dataset
# CUDA_VISIBLE_DEVICES=1 python test_few.py --mode few_shot --dataset mvtec \
# --data_path ./data/mvtec --save_path ./results/mvtec/few_shot/4shot/seed42 \
# --config_path ./open_clip/model_configs/ViT-L-14-336.json --checkpoint_path ./exps/pretrained/visa_pretrained.pth \
# --model ViT-L-14-336 --features_list 6 12 18 24 --few_shot_features 6 12 18 24 \
# --pretrained openai --image_size 518 --k_shot 4 --seed 42
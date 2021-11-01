cd ~/clip-alfred
conda activate alfred-1.5
export CUDA_VISIBLE_DEVICES=0
export ALFRED_ROOT=/home/ubuntu/clip-alfred

python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_4.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-5 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_9.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-10 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_14.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-15 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_19.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-20 (Valid Seen)"


cd ~/clip-alfred
conda activate alfred-1.5
export CUDA_VISIBLE_DEVICES=1
export ALFRED_ROOT=/home/ubuntu/clip-alfred
python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_4.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-5 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_9.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-10 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_14.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-15 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-rn50/net_epoch_19.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50 && echo "COMPLETED EXPERIMENT: ImageNet-20 (Valid Unseen)"

cd ~/clip-alfred
conda activate alfred-1.5
export CUDA_VISIBLE_DEVICES=2
export ALFRED_ROOT=/home/ubuntu/clip-alfred
python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_4.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-5 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_9.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-10 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_14.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-15 (Valid Seen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_19.pth --eval_split valid_seen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-20 (Valid Seen)"


cd ~/clip-alfred
conda activate alfred-1.5
export CUDA_VISIBLE_DEVICES=3
export ALFRED_ROOT=/home/ubuntu/clip-alfred
python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_4.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-5 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_9.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-10 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_14.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-15 (Valid Unseen)" && python models/eval/eval_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --model_path ../logs/alfred-baseline-clip/net_epoch_19.pth --eval_split valid_unseen --gpu --num_threads 3 --visual_model resnet50_clip && echo "COMPLETED EXPERIMENT: CLIP-20 (Valid Unseen)"

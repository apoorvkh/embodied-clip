ALFRED_ROOT=/home/ubuntu/clip-moca python models/train/train_seq2seq.py --data ../alfred_data/json_feat_2.1.0 --save_every_epoch --gpu --dout /home/ubuntu/logs/moca-clip --no_augmentation --visual_model resnet50_clip --batch 16 --epoch 30 --lr 1e-3 --vis_dropout 0.2 --hstate_dropout 0.2

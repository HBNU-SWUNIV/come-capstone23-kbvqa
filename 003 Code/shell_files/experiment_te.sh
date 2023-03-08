CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'all' --lang 'ko' --use_transformer_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'A' --lang 'ko' --use_transformer_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'B' --lang 'ko' --use_transformer_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'all' --lang 'en' --use_transformer_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'A' --lang 'en' --use_transformer_layer
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size=256 --n_epoch 31 --train_data 'B' --lang 'en' --use_transformer_layer
#!/usr/bin/env bash
#set desired experiment path
export EXP_DIR="runs/"
# Simple Embeddings
# Inner product simple 64
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version inner_prod --experiment_name cv_simple_inner --no_date --sample_weighting none --reg_constant 0.0 --embedding_version simple  --embed_size 64 --learning_rate 0.001 --verbosity 2 --n_epochs 40 --checkpoint --crossvalidate
# MLP1 simple 128 (observation: for MLP1 with simple embeddings 128 embedding size performs better than 64)
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version mlp1 --experiment_name cv_mlp1 --no_date --sample_weighting none --reg_constant 0.0 --embedding_version simple  --embed_size 128 --learning_rate 0.0001 --verbosity 2 --n_epochs 40  --crossvalidate
# MLP2 simple 64
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version mlp2 --experiment_name cv_mlp2 --no_date --checkpoint --sample_weighting none --reg_constant 0.0 --embedding_version simple  --embed_size 64 --learning_rate 0.0002 --verbosity 2 --n_epochs 100 --load_embedding_weights $EXP_DIR/cv_simple_inner --crossvalidate


# Autoencoder Embeddings
# Inner product Autoencoder 128
python train_embedding_based.py --out_dir $EXP_DIR --crossvalidate --experiment_name cv_ae_inner --no_date --embedding_version=autoencoder_embedding --learning_rate=0.001 --prediction_version=inner_prod --optimizer=adam --n_epochs=10 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/ --reg_constant=0.0001
# MLP1 Autoencoder 128
python train_embedding_based.py --out_dir $EXP_DIR --crossvalidate --experiment_name cv_ae_mlp1 --no_date --embedding_version=autoencoder_embedding --learning_rate=0.0001 --prediction_version=mlp1 --optimizer=adam --n_epochs=10 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/
# MLP2 Autoencoder 128
python train_embedding_based.py --out_dir $EXP_DIR --crossvalidate --experiment_name cv_ae_mlp2 --no_date --embedding_version=autoencoder_embedding --learning_rate=0.0001 --prediction_version=mlp2 --optimizer=adam --n_epochs=20 --embed_size=128 --load_embedding_weights_numpy=Weights/ae_embeddings/


# Attention Embeddings
# Inner product attention 64
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version inner_prod --experiment_name cv_attention_inner --no_date --sample_weighting none --reg_constant 0.0 --embedding_config config/attention.json  --embed_size 64 --n_context_movies_pred 128 --n_context_movies 64 --learning_rate 0.001 --verbosity 2 --n_epochs 40 --checkpoint --crossvalidate
# MLP1 attention 64
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version mlp1 --experiment_name cv_attention_mlp1 --no_date --sample_weighting none --reg_constant 0.0 --embedding_config config/attention.json  --embed_size 64 --learning_rate 0.0001 --verbosity 2 --n_epochs 70 --n_context_movies_pred 128 --n_context_movies 64 --load_embedding_weights $EXP_DIR/cv_attention_inner --crossvalidate
# MLP2 attention 64
python train_embedding_based.py --out_dir $EXP_DIR --prediction_version mlp2 --experiment_name cv_attention_mlp2 --no_date --sample_weighting none --reg_constant 0.0 --embedding_config config/attention.json  --embed_size 64 --learning_rate 0.0002 --verbosity 2 --n_epochs 40 --n_context_movies_pred 128 --n_context_movies 64 --load_embedding_weights $EXP_DIR/cv_attention_inner --crossvalidate

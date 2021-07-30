# Attentive Neural Collaborative Filtering

## Installation
using python 3.6 run

```bash
pip install numpy
pip install -r requirements.txt
```

Installing numpy first is necessary due to a [bug](https://github.com/NicolasHug/Surprise/issues/187) in the 'surprise' toolbox. 

## Data:

We first restructure the given csv files for easier use, such that there are four columns (sample_id,row_id,col_id,prediction).
The restructured csv files are provided in the data folder. They can be regenerated by:

```bash
python Utils/DataUtils.py --process_data --raw_csv data/data_train.csv --out data/structured_data_train.csv
python Utils/DataUtils.py --process_data --raw_csv data/sampleSubmission.csv --out data/structured_sample_submission.csv
```

Read the structured csv files with:

```python
import pandas as pd
data = pd.read_csv("data/structured_data_train.csv")
data.set_index("sample_id", inplace=True)
```
## Reproducing baselines results
```bash
python svd_final.py 
``` 
to run 10 fold cross validation using self implemented movie imputed SVD (reported in table 1)

For further data exploration and analysis using SVD refer ```Data_exploration.ipynb```

```bash
python surprise_baselines.py
``` 
to run 10 fold cross validation for the pre-imlpemented surprise baselines (reported in table 1)

## Reproducing the reported results

We provide a convenient bash script to reproduce all cross validation results reported in table 2 of our report. 
```bash
sh BashScripts/cross_validation_experiments.sh
```

Folders created for each of the experiments will contain a json file with the relevant metrics.  

**Note**, that this requires the autoencoder initial embeddings, which are provided in the Weights folder. To regenerate them use (while beeing in the toplevel folder of the submission folder)

```bash
export PYTHONPATH=$PWD/:$PYTHONPATH
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=user --train_data_path=data/ --embedding_save_path=Weights/ae_embeddings/
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=movie --act_hidden_e=tf.nn.sigmoid --n_hidden=128 --act_out_e=tf.nn.sigmoid --act_hidden_d=tf.nn.sigmoid --act_out_d=tf.nn.sigmoid --regularize=False --embedding_save_path=Weights/ae_embeddings/ --train_data_path=data/
```

# Usage of the General Embedding/Prediction Framework
In the following we detail the usage of our general framework for combining different embedding and prediction models.

The minimally required command is as follows
```bash
python train_embedding_based.py --embed_size [int] [embedding specification] [prediction specification] 
```

embedding and prediction models can be used with default parameters by just specifying the embedding/prediction_type as string.    
that is     
[prediction specification]= `--prediction_version [inner_prod | mlp1 | mlp2]`   
[embedding specification]= `--embedding_version [simple | autoencoder_embedding | attention]`

alternatively a json file can be passed that specifies additional parameters which will be given to the constructor of the model.   
e.g.   
[embedding specification]= `--embedding_config config/attention.json`

**Example call**
```bash
python train_embedding_based.py --embed_size 128 --embedding_config config/attention.json --prediction_version mlp2 
``` 

#### Aditional Parameters

`--batchsize [int]` : the batch size   
`--n_epochs [int]`  : how many epochs to train   
`--leanring_rate [float]` : the learning rate   
`--optimizer [adam| rms]` : which optimization algorithm to use      

`--force_gpu` : to force using the gpu     


`--out_dir [path]` : path to the folder in which the experiment should be saved      
`--exp_name [string]` : the name of the experiment   
`--no_date` : do not add timestamp to experiment name    
`--tensorboard [0 | 1]` : whether or not to write tensorboard summaries     

`--make_submission` : if this flag is present, a submission csv will be generated whenever the validation score reaches a new best      
`--checkpoint` : if this flag is present, checkpoints will be saved whenever the validation score reaches a new best     
`--crossvalidate` : if this flag is present, the experiment will carry out 10-fold cross validation using the fixed split defined in `data/s10.pickle`. (We rely on a fixed split that we generated once before doing any experiments)

`--load_embedding_weights [path]` and   
`--load_prediction_weights [path]`   
for loading pretrained weights of the embedding and prediction model (possibly separately). You can for example pretrain the attention embedding using the inner product prediction model and then further fine tune with an MLP variant. 
the `[path]` has to be the path to the experiment folder (e.g. `$EXP_DIR/2019_07_01-08_55_55test)`.   
Pretrained weights can be used with `--crossvalidate`, as long as the pretraining was also done with `--crossvalidate`. The correct splitting will be observed.

**Specific to attention embedding**   
`--n_context_movies [int]` : the number of previously watched movies that are sampled for a user to generate the embedding during training   
`--n_context_movies_pred [int]` : the same during testing. 

**Specific to autencoder embedding**   
`--load_embedding_weights_numpy=Weights/ae_embeddings/` has to be given 
 

## Autoencoder Embeddings:

Training the autoencoder-prediction model is so far only possible with pretrained autoencoder embedidngs.
To reconstruct the results for AE embeddings in the prediction models in the report, first the AE embeddings need to be saved for each split, then cross validation needs to be run with the prediction model (mlp1, mlp2 or inner_product).

### Train autoencoder alone

The autoencoder can be trained with the following command:

Set the directory: `export PYTHONPATH=$PWD/:$PYTHONPATH`

User autoencoder: `python -u autoencoder/autoencoder.py --embedding=user --train_data_path=data/`

Movie autoencoder: `python -u autoencoder/autoencoder.py --embedding=movie --train_data_path=data/`

Note: The movie autoencoder is simply trained by inverting the matrix, such that items become users and users become items.

Path flags to set:
* --save_path (default: '../Weights/model3/model'): Path to save checkpoints
* --summaries_dir (default: 'logs') : Output dir for tensorboard
* --train_data_path (default: '../data'): Path where the structured_data_train.csv file is located
* --val_part (default: '9'): Which part of the predefined split should be used as validation?

Network specific flags to set:

* --act_hidden_d (default: 'tf.nn.relu')
* --act_hidden_e (default: 'tf.nn.relu')
* --act_out_d (default: 'None')
* --act_out_e (default: 'tf.nn.relu')
* --batch_size (default: '8')
* --dropout_rate  (default: '1.0')
* --epochs (default: '200')
* --regularize (default: 'true')
* --lamba (default: '0.001')
* --n_embed (default: '128')
* --n_hidden (default: '512')
* --learning_rate (default: '0.0001')

### Get autoencoder embeddings:

Run the same command as above, but set the flag --CV_save_embed=True. This will train the model for each split, and save the trained embeddings in the specified save path (--embedding_save_path (default: '../Weights/ae_embeddings/')). These weights were user together with the prediction model to get to the result stated in the report.

To save the embeddings per fold, run:

1) **user** autoencoder:
```bash
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=user --train_data_path=data/ --embedding_save_path=Weights/ae_embeddings/
```
(The default parameter values are set such that they are optimal for the user autoencoder, so no network-specific flags need to be set here)

2) **movie** autoencoder:
```bash
python autoencoder/autoencoder.py --epochs=100 --CV_save_embed=True --embedding=movie --act_hidden_e=tf.nn.sigmoid --n_hidden=128 --act_out_e=tf.nn.sigmoid --act_hidden_d=tf.nn.sigmoid --act_out_d=tf.nn.sigmoid --regularize=False --embedding_save_path=Weights/ae_embeddings/ --train_data_path=data/
```

### Other autoencoder variants:
Another approach was to train two autoencoders simultaniously, computing the loss as the sum over both AE reconstruction losses as well as the MSE of a rating that was yielded with a short FFNN. To run this network:

```bash
cd autoencoder
python comb_ae.py
```

It will show train and validation scores, and you can interrupt at any point and predict on the test data. See flags for possible parameters, most importantly specify --save_path (where to output submission) and --sample_submission_path (where sampleSubmission file is located).




# Overview 



## preprocessing

- Modify the ``save_root`` and ``src_root`` in ``./preprocessing/condi-seg/00.baseline_data_clean.py`` and run as follows. It will convert the nifty images into numpy array and save them in the ``save_root``
```
python 00.baseline_data_clean.py
```


## train 

- Examples of the commandlines for running the 9 experiments in the excel table are listed in ``./scripts/condi-seg/``. Remember to change the data path in each of the scripts.

- If run on local servers/PCs (Not cs cluster), use ``sh ./scripts/condi-seg/xxx.sh`` to train a model. But need to set ``--use_HPC 0`` and  ``--gpu gpu_id``. Otherwise the model will only train on the GPU 0.

- If train on cluster. Use the following example command lines if need to submit multiple files. This tool will submit the scripts if its name mactches the regular expressions.
```
python ./Tools/hpc/qsub.py ./scripts/condi-seg/hpc.01\*
```


## test

- After the training is done, you can copy the models back and do the inference in a local machine to avoid queuing. 
- The models will be saved in ``./logs/[--project]/[--exp_name]``
- Use the following command to do the inference. It will do inference on the best model if the ''[num_epoch]'' is omit.
```
python test.py ./logs/[--project_name]/[--exp_name] [gpu_id] [num_epoch]
```

# DVAE_torch

*Note this code is research code used to test a models performance on a benchmark dataset. This is not optimized or designed for production/industry applications. I simply post it here as it may help 
others understand my research or the topics presented here. 

This repository contains the code for training and testing Dynamical Variational Autoencoders (DVAEs) using PyTorch. Specifically, this is applied to the CMAPSS turbofan engine dataset from NASA
(data can be found in this repository or [here](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) under header 6. Turbofan Engine Degradation Simulation). 
Here it is applied to estimate the Remaining Useful Life (RUL) of each turbofan engine given the sensors from the engine. This is based on work done in Chapter 5 of my PhD thesis 
"Degradation Vector Fields with Uncertainty Considerations" (which will soon be available). 

The training and testing scripts can be run with various arguments to train and test the supervised model for RUL estimation. The semi_supervised script trains a semi-supervised model 
by splitting the training data further and hiding a portion of the RUL targets from the model to test how well the semi-supervised model can perform with small amounts of target data available. 
The testing script can also test trained semi-supervised models using the "--semisupervised True" argument. 

There is also 2 notebooks in the notebooks directory. One notebook shows how to construct a DVAE model and use it for RUL estimation, which is hopefully less abstract and 
easier to follow than the training and testing scripts that have to accommodate many different possible arguments. 
The semi-supervised notebook also goes over a semi-supervised model construction and testing case and is also the notebook used prior to coding the script version 
(so it functions as a sort of prototype). 

The key insight here is to use the sensors as conditional variables in a conditional DVAE model, and the RUL are the input to the DVAE model. If this is the case then the conditional variables 
should be represented as noncausal sequences,

$$
p(y_{1:T}|x_{1:T}) = \int p(y_{1:T}, z_{1:T} | x_{1:T}) dz_{1:T}
$$

The term in the integral can be expanded into 

$$
p(y_{1:T}, z_{1:T} | x_{1:T}) = \prod_{t=1}^T p(y_t|y_{1:t-1}, z_{1:t}, x_{1:T}) p(z_t|z_{1:t-1}, y_{1:t-1}, x_{1:T})
$$

Notice if $y_{1:T}$ is our input sequence (the RUL sequence in this case) and $x_{1:T}$ is our conditional sequence (our sensor observation sequences in this case) then $x_{1:T}$ does 
not simplify further and must be kept noncausal (current time variables such as $z_t$ and $y_t$ rely on past, present and future variables). A sequence-to-sequence model is used
to implement this practically for RUL estimation. If this is done, then the DVAE is trained to sample from the distribution $p(y_{1:T}|x_{1:T})$ which represents the probability of 
the RUL given the sensor observations. Hence, this allows us to train a probabilistic RUL estimation model. 

training.py 
  The arguments:
```
  --dataset chooses between the CMAPSS dataset options "FD001", "FD002", "FD003", "FD004"
  --save_path type in a string such as "saved_models/DVAE" the file "saved_models" already exists for you to save models in. This saves files during training based on the validation loss
    Note this is just the beginning of the file name the full file name is "save_path_transition_measurement_inference_encoder_initializer_dataset.pth
    Notice each of those terms are arguments in this training script, and they are all strings, so if you change any of these, a different save file is made.
  --zdim is the dimension size of the latent variables 
  --hdim is the dimension size of the hidden variables of the various networks used in the DVAE 
  --T is the sliding window size for the sequence-to-sequence models 
  --lr is the learning rate used during training 
  --L2 is the weight_decay value used in the optimizer 
  --bs is the batch size used for the training dataloader 
  --stride is the stride used when preparing the training and validation data
  --split is the training/validation split e.g. 0.2 means 80/20 training/validation split 
  --epochs is the number of epochs used in training 
  --transition allows the user to choose various transition models "mlp" is the only option but more may be added in the future 
  --measurement chooses the measurement model, agian only "mlp" is available 
  --inference chooses the inference model used for the DVAE, only "rnn" is available 
  --encoder chooses the encoder that encoders the causal and noncausal sequences such as $y_{1:t}$ and $x_{1:T}$, "rnn" is the best option although "none" exists (mainly for other cases such as unsupervised learning)
  --initializer chooses the initializer method for calculating the initial latent variable $z_0$ "controls" applies a feedforward network on the first control variable and the output is $z_0$
  --transition_inputs a string that allows the user to control the conditions of the transition model distribution e.g. "zyx" gives $p(z_t|z_{1:t-1}, y_{1:t-1}, x_{1:T}$ while "zx" simplifies it to $p(z_t|z_{1:t-1}, x_{1:T}$
  --measurement_inputs does the same as transition_inputs but for the measurement model 
  --init_inputs does the same as the other _inputs arguments, however, currently the options for initilizer models mean this doesn't actually affect their models. 
  --controls if True this creates a conditional model otherwise if we want to train an unsupervised DVAE model (maybe to reconstruct the sensor inputs instead of RUL estimation) then we can set this to False 
  --unsupervised if True then we train an unsupervised model (used to train a DVAE for sensor reconstruction not RUL estimation) 
  --load_hyperparameters if you have some saved hyperparameters in the "saved_hypes" directory (see the hyperparameters directory on how to perform Bayesian Optimization to populate that file) 
  --trial is the trial that should be loaded from the hyperparameters file if load_hyperparameters was set to True
```
testing.py 
  Has many similar arguments as the training.py script for model construction. Note you want to have the same "save_path" argument as training.py as the testing.py script will load the model that was trained 
  and saved using this script. The arguments are as follows:
  ```
  --dataset is the same as in training.py 
  --save_path same as in training.py 
  --run_model if you have already run the testing script but you want to generate some of the plots again without having to wait for the model to revaluate the entire testing dataset again set this to True 
  as the results of the testing script are saved so they can just be loaded without having to run the model again. 
  --plot_unit set to False if you don't want to see the plots
  --transition same as in training.py, this determines which saved model file is loaded
  --measurement same as in training.py, this determines which saved model file is loaded
  --inference same as in training.py, this determines which saved model file is loaded 
  --encoder same as in training.py, this determines which saved model file is loaded
  --initializer same as in training.py, this determines which saved model file is loaded
  --best if any of the files have the string "best" after the save_path string in the file name, that file is chosen for evaluation (this is if you want to test the same type of model with different 
  hyperparameters but don't want to lose the file with a model that performs well, manually change the file to have this format and you can evaluate it with this script without needing to change it back manually)
  --controls same as in the training.py script 
  --unsupervised same as in the training.py script 
  --unit some plots will provide a plot for a specific testing unit, this lets you choose which one
  --split used for the semi-supervised model testing, this is the split that artificially gets rid of some of the targets from the training set. For the semi-supervised models this is part of their file 
  names and is used here to choose that file for testing. 
  --semisupervised set this to True if you want to test a trained semi-supervised model 
  --N is the number of samples generated when testing, the more there are the closer the model approximates $p(y_{1:T}|x_{1:T})$
```
semi_supervised.py
```
  It has the same arguments as training.py but also adds some additional ones, such as:
  --split is now an int that determines how many targets to remove from the training dataset. e.g. 90 gets rid of 90% of the targets so the semi-supervised model is required to train a model for RUL estimation 
  using only 10% of the RUL targets that would normally be available for the fully supervised model.
  --valid_split is what split was in the training.py script e.g. 0.2 would mean a 80/20 training/validation split 
  *Note all these pre_ arguments are effectively the same as their counterparts without the pre_ prefix. However, these apply for defining the unsupervised model used in the semi-supervised training setup. 
  --pre_transition allows the user to choose various transition models "mlp" is the only option but more may be added in the future 
  --pre_measurement chooses the measurement model, agian only "mlp" is available 
  --pre_inference chooses the inference model used for the DVAE, only "rnn" is available 
  --pre_encoder chooses the encoder that encoders the causal and noncausal sequences such as $y_{1:t}$ and $x_{1:T}$, "rnn" is the best option although "none" exists (mainly for other cases such as unsupervised learning)
  --pre_initializer chooses the initializer method for calculating the initial latent variable $z_0$ "controls" applies a feedforward network on the first control variable and the output is $z_0$
  --pre_transition_inputs a string that allows the user to control the conditions of the transition model distribution e.g. "zyx" gives $p(z_t|z_{1:t-1}, y_{1:t-1}, x_{1:T}$ while "zx" simplifies it to $p(z_t|z_{1:t-1}, x_{1:T}$
  --pre_measurement_inputs does the same as transition_inputs but for the measurement model 
  --pre_init_inputs does the same as the other _inputs arguments, however, currently the options for initilizer models mean this doesn't actually affect their models. 
```

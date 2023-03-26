---- [data] contains data files (training data, testing data, trained model files) for running the jupyter notebook files
      |
      ---- [EnLFT] contains trained EnLFT model, the training data and testing data for training the model
      |
      ---- [LFT] same data files for the LFT model
      |
      ---- [MF] same data files for the MF model
      |
      ---- [PC] same data files for the PC-MF model
      |
      ---- [VAE] same data files for the VAE model
|
---- [models] contains python files for training the models, all from github repo of Ziwei
|
---- [scripy_MF.ipynb] the jupyter notbook file for calculating all metrics (activeness, popolarity, ......) of the MF model
|		  !!!(This file contains the detailed comments)
|
---- [scripy_EnLFT.ipynb] the jupyter notbook file for calculating all metrics of the EnLFT model
|
---- [scripy_LFT.ipynb] the jupyter notbook file for calculating all metrics of the LFT model
|
---- [scripy_PC_MF.ipynb] the jupyter notbook file for calculating all metrics of the PC_MF model
|
---- [scripy_VAE.ipynb] the jupyter notbook file for calculating all metrics of the VAE model


!!! in the [data] folder, all the sub-folders(EnLFT, VAE ......) have the same training and testing data files because we train all the models by the exactly same data
!!! the file [script_MF.ipynb] is the first written file, so it contains the most detailed comments, all other scripts files for other models are copied and modified from this file
!!! the python files in the [models] folder cannot be run directly, but to make the results reproducible, the parameters (hypterparameters) can be found in them 
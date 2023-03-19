This directory contains Prak acoustic model training and inference (and trained model).
It also contains Praat TextGrid files i/o.

You do not have to run any training if you just want to use the NN AM for alignment.
If you want to repeat the basic training yourself using just the publicly available data,
run these Jupyter notebooks:
* [`Prepare_Training_Data.ipynb`](/acmodel/Prepare_Training_Data.ipynb)
* [`NN_Train_Align.ipynb`](/acmodel/NN_Train_Align.ipynb)

If you have additional manually fine tuned TextGrid files, you can train a tuned model. Run also these:
* [`Prep_Manual_Train_Data.ipynb`](/acmodel/Prep_Manual_Train_Data.ipynb)
* [`NN_Train_Align_Man.ipynb`](/acmodel/NN_Train_Align_Man.ipynb)

Notebooks run for up to 50 iterations, you can interrupt the training much earlier (like after 20).

# PEGASOS
Gradient based solver for SVM

This is a multi-class classifier. The classification was done on the Fashion-MNIST dataset, which has 10 classes. 

The accuracy achieved was 80.2%. The confusion matrix is in the report. The sklearn-library achieved an accuracy 80.37% on the same dataset.

#### Requirements:

Python3, Linux

* pandas
* scipy
* numpy
* sklearn
* seaborn
* matplotlib

#### Running the classifier:

To run the code and replicate results:

1.	Clone the fashion mnist repository using
	git clone https://github.com/zalandoresearch/fashion-mnist.git

2.	Copy the code (pegasosSVM.py) into the fashion mnist directory

3.	Run "python3 pegasosSVM.py" in the terminal. It takes around 10-13 minutes to execute

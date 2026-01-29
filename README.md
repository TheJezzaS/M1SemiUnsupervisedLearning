Semi-Supervised Learning with Deep Generative Models (M1)

This project implements the M1 semi-supervised learning model from Kingma et al. using a Variational Autoencoder (VAE) for feature extraction and an SVM classifier for prediction. The model is evaluated on the Fashion-MNIST dataset.

Training is performed in two stages. First, the VAE is trained in an unsupervised manner on the full Fashion-MNIST training set. After training, the encoder mean vector is used as a latent representation for each image and the VAE weights are saved to disk.

Second, an SVM with an RBF kernel is trained on latent representations extracted from a balanced labeled subset of the training data. Experiments are conducted using 100, 600, 1000, and 3000 labeled samples. A fixed random seed is used for reproducibility. Each trained SVM model is saved to disk.

To train and test the model, run:

python main.py

The script trains the VAE, trains one SVM per label setting, saves all models, and reports classification accuracy on the Fashion-MNIST test set.
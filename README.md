# Flower Image Classifier - Deep Learning

In this project a deep learning network is trained to identify flowers in images. The bottom of the HTML/ipynb shows the results on various photos taken around London - I've no idea if that are correct or not!

After creating a pre-processing pipeline, a MobileNet pre-trained network from TensorFlow Hub is loaded and used to create a new feed-forward network as a classifier. Following training using a GPU, the classifier achieves 77% accuracy on the Oxford flower dataset. Finally, the model is saved and is used as an input to the python command line application (run "python predict.py -h" for usage).


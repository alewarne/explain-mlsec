# Network Training

### This folder contains Training scripts for the four neural networks based on the papers:

* Adversarial examples for malware detection (Grosse et al.) - called Drebin.
* Deep Android Malware Detection (McLaughlin et al.) - called DAMD.
* VulDeePecker: A Deep Learning-Based System for Vulnerability Detection (Li et al.) - called VulDeePecker.
* LEMNA: Explaining Deep Learning based Security Applications (Guo et al.) - called Mimicus.

#### To keep the size of this repo handable, we do not deliver all the datasets with it but all of them are accessible for download online. Each folder contains a config file where you can adjust training parameters.
* Mimicus: Call `python3 mimicus.py` to train the network.
* Drebin: Adjust the paths in the config file to point to the location you downloaded the [drebin dataset](https://www.sec.cs.tu-bs.de/~danarp/drebin/) to. Call `drebin.py` to train the network.
* VulDeePecker: Extract the json file from the zip. Afterwards run `python3 word2vec.py` to train a word2vec model. Then run `python3 vuldeepecker.py` to train the network.
* DAMD: Extract the folder containig the dalvik opcodes. Afterwards run `python3 preprocessing.py` to convert them. Then run `python3 damd.py` to train the network.

#### The models with the best performance will be saved in the models folder.
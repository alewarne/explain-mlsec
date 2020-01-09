# these paths actually come with the drebin dataset
doc_path = '' # insert path to feature vectors folder from drebin dataset
json_path = '' # insert path to virustotal.json from drebin dataset
split_path = '' # insert path to datasplits/all from drebin dataset

# parameters for learning
threshold = 10  # at least 10 scanners have to exist and classify the sample as malicious
batch_size = 64  # batch_size during training
split_index = 0  # which of the data splits from drebin dataset to use (0,..,10)
vec_output = True  # whether to output vectors [1,0], [0,1] or float at the end of the network
nonlinearity = 'softmax'  # nonlinearity in the final layer
loss = 'binary_crossentropy'  # loss for training
epochs = 50  # number of training epochs

# implementation of TPR and FPR with keras backend in order to access them at the end of each learning epoch as a metric
# on the test set

import keras.backend as K


# calculates true positive metric using keras backend. predictions y_hat will be normalized and do not need to be a
# probability distribution. the vec_output parameter specifies whether the output/labels are one-hot encoded or
# one dimensional (which is only possible in a binary classification problem)
def custom_true_positive_metric(vec_output):

    def true_positive_metric(y, y_hat):
        y_hat_rounded = K.round(y_hat)
        if vec_output:
            ground_truth_labels = K.cast(K.argmax(y), dtype='float32')
            predicted_labels = K.cast(K.argmax(y_hat_rounded), dtype='float32')
        else:
            ground_truth_labels = K.cast(y, dtype='float32')
            predicted_labels = K.cast(y_hat_rounded, dtype='float32')
        ground_truth_equal_one = K.cast(K.equal(K.ones(K.shape(ground_truth_labels)), ground_truth_labels), dtype='float32')
        prediction_equal_one = K.cast(K.equal(K.ones(K.shape(predicted_labels)), predicted_labels), dtype='float32')
        # product of these two vectors is 1 if and only if both conditions are met. Sum the product to get the number of samples
        nominator_TPR = K.sum(ground_truth_equal_one * prediction_equal_one)
        denominator_TPR = K.sum(ground_truth_equal_one)
        return nominator_TPR / (denominator_TPR)#+K.epsilon())

    return true_positive_metric


def custom_false_positive_metric(vec_output):

    def false_positive_metric(y, y_hat):
        y_hat_rounded = K.round(y_hat)
        if vec_output:
            ground_truth_labels = K.cast(K.argmax(y), dtype='float32')
            predicted_labels = K.cast(K.argmax(y_hat_rounded), dtype='float32')
        else:
            ground_truth_labels = K.cast(y, dtype='float32')
            predicted_labels = K.cast(y_hat_rounded, dtype='float32')
        ground_truth_equal_zero = K.cast(K.equal(K.zeros(K.shape(ground_truth_labels)), ground_truth_labels), dtype='float32')
        prediction_equal_one = K.cast(K.equal(K.ones(K.shape(predicted_labels)), predicted_labels), dtype='float32')
        # product of these two vectors is 1 if and only if both conditions are met. Sum the product to get the number of samples
        nominator_FPR = K.sum(ground_truth_equal_zero*prediction_equal_one)
        denominator_FPR = K.sum(ground_truth_equal_zero)
        return nominator_FPR / (denominator_FPR)#+K.epsilon())

    return false_positive_metric

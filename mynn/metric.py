import cupy as cp

def accuracy(preds, labels):
    """
    Metric for MNIST.
    preds : [batch, D]
    labels : [batch, ]
    """
    assert preds.shape[0] == labels.shape[0]

    predict_label = cp.argmax(preds, axis=-1)
    
    return (predict_label == labels).sum() / preds.shape[0] 
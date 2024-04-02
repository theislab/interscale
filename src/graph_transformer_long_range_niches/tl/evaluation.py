def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)
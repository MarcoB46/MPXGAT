import numpy as np

class EarlyStopper:
    """
    Perform early stopping if validation loss does not improve for a certain number of epochs.
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    def __init__(self, patience=1, delta=0, delta_decay=0.001, min_delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.delta_decay = delta_decay
        self.min_delta = min_delta

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def decrease_delta(self):
        self.delta -= self.delta_decay
        if self.delta < self.min_delta:
            self.delta = self.min_delta

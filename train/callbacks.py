from torchmetrics import Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt
import pickle


class History:
    def __init__(self, data=None):
        if data == None:
            self.history = {"train": {}, "val": {}}
        else:
            self.history = data

    def update(self, metrics, train = 'train'):
        for key, value in metrics.items():
            if key not in self.history[train]:
                self.history[train][key] = []
            self.history[train][key].append(value)

    def plot(self):
        # plot three metrics
        epochs = range(1, len(self.history['val']['loss']) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, (key, _) in enumerate(self.history['val'].items()):
            if (i > 2):
                break
            axs[i].plot(self.history['train'][key][:len(epochs)], label='train')
            axs[i].plot(self.history['val'][key][:len(epochs)], label='val')
            axs[i].set_title(key)
            axs[i].set_xlabel('Epochs')
            axs[i].legend()
        plt.show()
        
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.history, file)



class Metrics:
    def __init__(self, metrics):
        self.metrics = metrics

    def update(self, y_true, y_pred):
        for _, metric in self.metrics.items():
            metric.update(y_pred, y_true)

    def compute(self):
        return {key: metric.compute() for key, metric in self.metrics.items()}

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()


def def_metrics(device):
    return Metrics({
        'accuracy': Accuracy(task='binary').to(device),
        'precision': Precision(task='binary').to(device),
        'recall': Recall(task='binary').to(device),
        'f1score': F1Score(task='binary').to(device)
    })

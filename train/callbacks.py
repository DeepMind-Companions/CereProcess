from torchmetrics import Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import pickle
import os

def get_edf_file(file, path):
    # remove .npz and replace with .edf from file
    file = file.replace('.npz', '.edf')
    return os.path.join(path, file)

class History:
    def __init__(self, data=None):
        if data == None:
            self.history = {"train": {}, "val": {}}
        else:
            self.history = data
        self.best = {
            "loss": {
                "loss": 1000.0,
            }, 
            "accuracy": {
                "accuracy": -1.0,
            }, 
        }
        self.file_preds = []
        self.cm = {
            "actual": [],
            "predicted": []
        }

    def new_file_preds(self):
        self.file_preds.append([])

    def update_file_preds(self, file, target, output):
        self.file_preds[-1].append((file, target, output))

    def get_file_preds(self):
        result = {
            "FP":[],
            "FN":[],
            "TP":[],
            "TN":[]
        }
        for file, target, output in self.file_preds[-1]:
            if torch.argmax(target) == 1:
                if torch.argmax(output) == 1:
                    result["TP"].append((file, target, output))
                else:
                    result["FN"].append((file, target, output))
            else:
                if torch.argmax(output) == 1:
                    result["FP"].append((file, target, output))
                else:
                    result["TN"].append((file, target, output))
        
        return result

    def _update_confidence(self, file, result, confidence):
        if result <= 0.2:
            confidence[0].append(file)
        elif result <= 0.4:
            confidence[1].append(file)
        elif result <= 0.6:
            confidence[2].append(file)
        elif result <= 0.8:
            confidence[3].append(file)
        else:
            confidence[4].append(file)


    def get_confidence(self, typ="ALL"):
        result = self.get_file_preds()
        # Divide the results into confidence rate of 20%
        # 0-20% 20-40% 40-60% 60-80% 80-100%
        # TP, FP, TN, FN
        confidence = {
            "TP": [[], [], [], [], []],
            "FP": [[], [], [], [], []],
            "TN": [[], [], [], [], []],
            "FN": [[], [], [], [], []]
        }
        # TP
        for file, _, output in result["TP"]:
            clss = torch.max(output)
            self._update_confidence(file, clss, confidence["TP"])
        for file, _, output in result["TN"]:
            clss = torch.max(output)
            self._update_confidence(file, clss, confidence["TN"])
        for file, _, output in result["FP"]:
            clss = torch.max(output)
            self._update_confidence(file, clss, confidence["FP"])
        for file, _, output in result["FN"]:
            clss = torch.max(output)
            self._update_confidence(file, clss, confidence["FN"])

        if (typ == "ALL"):
            return confidence
        else:
            try:
                return confidence[typ]
            except:
                raise ValueError("Invalid type")

    def plot_confidence(self, typ):
        confidence = self.get_confidence()
        counts = [len(category) for category in confidence[typ]]
        categories = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        plt.bar(categories, counts)
        # Add titles and labels
        plt.title('Number of Files by Confidence Level')
        plt.xlabel('Confidence Range')
        plt.ylabel('Number of Files')
        plt.show()
    
    def update_cm(self, actual, pred):
        self.cm["actual"] = actual
        self.cm["pred"] = pred

    def display_cm(self):
        conf = confusion_matrix(self.cm["actual"], self.cm["pred"])
        cm = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=['normal', 'abnormal'])
        cm.plot(cmap = plt.cm.Blues)

    def update(self, metrics, train = 'train'):
        for key, value in metrics.items():
            if key not in self.history[train]:
                self.history[train][key] = []
            self.history[train][key].append(float(value))
        
        if train == 'val':
            if self.best["loss"]["loss"] > float(metrics["loss"]):
                for key, value in metrics.items():
                    self.best["loss"][key] = float(value)
            if self.best["accuracy"]["accuracy"] < float(metrics["accuracy"]):
                for key, value in metrics.items():
                    self.best["accuracy"][key] = float(value)

    def print_best(self):
        print("\nPrinting Best:")
        print("Loss Wise:")
        print(f"Best Loss: {self.best['loss']['loss']}")
        print(self.best["loss"])
        print(f"Best Accuracy: {self.best['accuracy']['accuracy']}")
        print(self.best["accuracy"])
            



    def plot(self, items = None):
        # plot three metrics
        epochs = range(1, len(self.history['val']['loss']) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
       # for i, (key, _) in enumerate(self.history['val'].items()):
       #     if (i > 2):
       #         break
       #     axs[i].plot(self.history['train'][key][:len(epochs)], label='train')
       #     axs[i].plot(self.history['val'][key][:len(epochs)], label='val')
       #     axs[i].set_title(key)
       #     axs[i].set_xlabel('Epochs')
       #     axs[i].legend()
        if items is None:
            items = ['loss', 'f1score', 'accuracy']
        else:
            if len(items) > 3:
                items = items[:3]
        for i, key in enumerate(items):
            axs[i].plot(self.history['train'][key][:len(epochs)], label='train')
            axs[i].plot(self.history['val'][key][:len(epochs)], label='val')
            axs[i].set_title(key)
            axs[i].set_xlabel('Epochs')
            axs[i].legend()
        plt.show()
        
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)



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

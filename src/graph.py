import matplotlib.pylab as plt
import numpy as np
from pathlib import Path


class Plotter():
    def __init__(self, title):
        project_root_dir = Path(__file__).parents[1]
        self.path_to_save = str(project_root_dir) + "/plotted-graphs/"

        self.title = title

        plt.ylim(top=1)  # adjust the top leaving bottom unchanged
        plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
  

        self.train_loss_data = []
        self.val_loss_data = []
        self.val_acc_data = []
        self.graph_in_time = []

    def add_data(self, train_data, valloss_data, valacc_data, in_time, highest_acc_list):

        self.highest_acc = highest_acc_list[1]
        self.highest_acc_time = highest_acc_list[0]

        self.train_loss_data.append(train_data)
        self.val_loss_data.append(valloss_data)
        self.val_acc_data.append(valacc_data)
        self.graph_in_time.append(in_time)

    def plot_graph(self):

        plt.plot(self.graph_in_time, self.train_loss_data, label = "Trainings loss", color="blue")
        plt.plot(self.graph_in_time, self.val_loss_data, label = "Validation loss", color="red")
        plt.plot(self.graph_in_time, self.val_acc_data, label = "Validation Accuracy", color ="green"  )
        

        plt.title(self.title+": vall accuracy: {:.4f} at epoch {:.2f}".format(self.highest_acc, self.highest_acc_time))

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        

        plt.savefig(self.path_to_save+self.title+".png")

        

        
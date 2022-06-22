from optimizer import SGD, AdaGrad, Adam
import numpy as np
import os
import csv


class Trainer:
    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=1,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
        eval_sample_num_per_epoch=None,
        verbose=True,
    ):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.eval_sample_num_per_epoch = eval_sample_num_per_epoch

        optimizer_dict = {"sgd": SGD, "adagrad": AdaGrad, "adam": Adam}
        self.optimizer = optimizer_dict[optimizer.lower()](**optimizer_param)
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        self.result = {}
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.mini_batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("Train loss: " + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.eval_sample_num_per_epoch is None:
                t = self.eval_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(
                    "=== Epoch: "
                    + str(self.current_epoch)
                    + " , Train acc"
                    + str(train_acc)
                    + ", Test acc:"
                    + str(test_acc)
                    + "==="
                )

            self.current_iter += 1

    def save_result(self, file_name="result.csv"):
        if not os.path.exists("./result"):
            os.makedirs("./result")
        file_name = os.path.join("./result", file_name)
        with open(file_name, "wb") as f:
            writer = csv.writer(f)
            writer.writerow(self.result.keys())
            writer.writerows(self.result.values())

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)
        self.test_acc_list.append(test_acc)
        self.result["train_loss"] = self.train_loss_list
        self.result["train_acc"] = self.train_acc_list
        self.result["test_acc"] = self.test_acc_list
        self.network.save_parameter()
        self.save_result()
        if self.verbose:
            print("=== Final Accuracy: " + str(test_acc))

from dataset import load_mnist
from model import SimpleConvNet
from trainer import Trainer

x_train, t_train, x_test, t_test = load_mnist(flatten=False)
epochs = 10

model = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01,
)

trainer = Trainer(
    model,
    x_train,
    t_train,
    x_test,
    t_test,
    epochs=epochs,
    mini_batch_size=1000,
    optimizer="Adam",
    optimizer_param={"lr": 0.01},
    eval_sample_num_per_epoch=1,
)


if __name__ == "__main__":
    trainer.train()

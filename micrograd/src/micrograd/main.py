from tqdm import tqdm

from micrograd.nn import MLP

EPOCHS = 10000
LR = 0.1


def main():
    x = [1.0, 2.0, 3.0]
    network = MLP(3, [10, 10, 5])
    labels = [1.0, 1.0, 1.0, 0.0, 0.0]

    for i in tqdm(range(EPOCHS)):
        pred = network(x)
        loss = network.mse_loss(pred=pred, labels=labels)
        print(loss)
        network.zero_grad()

        loss.backward()

        for p in network.parameters():
            p.data += -LR * p.grad

        print(f"loss: {loss} - {pred}")

    print(f"target: {labels}")


if __name__ == "__main__":
    main()

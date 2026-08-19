from dl_from_scratch.nn import MLP


def main():
    x = [1.0, 2.0, 3.0]
    network = MLP(3, [10, 10, 5])

    out = network(x)

    print(out)
    print(network.parameters())


if __name__ == "__main__":
    main()

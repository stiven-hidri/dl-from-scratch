from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "names.txt"


def main():
    with open(DATA_PATH) as f:
        name = f.read().splitlines()
    print(name[10])


if __name__ == "__main__":
    main()

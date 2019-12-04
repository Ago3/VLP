from vlp import main, vqa_main, train_main
import argparse


def run():
    parser = argparse.ArgumentParser()
    # main(parser)
    # vqa_main(parser)
    train_main(parser)


if __name__ == '__main__':
    run()

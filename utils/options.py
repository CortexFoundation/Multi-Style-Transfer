import argparse
import os

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for MXNet-Gluon-Style-Transfer")
        self.parser.add_argument("--config_path", type=str,
                                    help="Path of config file")

    def parse(self):
        return self.parser.parse_args()

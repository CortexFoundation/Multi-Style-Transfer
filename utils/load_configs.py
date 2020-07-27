import os
import yaml


class Configs(object):
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self._load_cfg()

    def _load_cfg(self):
        assert os.path.isfile(self.cfg_path), 'InputError: {} is not a file'.format(self.cfg_path)
        print('Loading configs from {}'.format(self.cfg_path))
        with open(self.cfg_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in cfg.items():
                setattr(self, key, value)

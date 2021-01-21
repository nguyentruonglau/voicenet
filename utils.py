import numpy as np
import configparser as ConfigParser
from optparse import OptionParser

    
def read_config_train():
    """Read config file part train

    Args:
        cfg_file (string): path to config file

    Returns:
        [Namespace object]: containing the arguments to the command
    """

    parser = OptionParser()
    parser.add_option("--cfg") #mandatory
    (options, args) = parser.parse_args()
    cfg_file = options.cfg
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)

    options.img_height = Config.get('train', 'img_height')
    ooptions.img_width = Config.get('train', 'img_width')
    options.image_size = Config.get('train', 'image_size')
    options.batch_size = Config.get('train', 'batch_size')
    options.seed = Config.get('train', 'seed')
    options.init_lr = Config.get('train', 'init_lr')
    options.path_save_model = Config.get('train', 'path_save_model')
    options.validation_split = Config.get('train', 'validation_split')
    
    return options
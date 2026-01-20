import argparse
import os
import os.path as osp
import random
import sys
import time
import distutils.util
import numpy as np
import torch
import os,sys
os.chdir(sys.path[0])

sys.path.append("./")

PAD = 0
UNK = 2
BOS = 3
EOS = 1

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

DATASETS = ["twitter", "memetracker", "android", "christianity"]

# timescales
TOSECOND = 1
TOMINUTE = 60 * TOSECOND
TOHOUR = 60 * TOMINUTE
TODAY = 24 * TOHOUR
TOWEEK = 7 * TODAY
TOMONTH = 4 * TOWEEK
CURRENT_TIME = int(time.time())
SEED = 2025

DEVICE = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def SeedEverything(SEED):
    
    print(f"Init_every_seed with {SEED}")

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    np.random.seed(SEED)
    np.set_printoptions(threshold=np.inf)
   
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()


# data paths for the model
class Options(object):
    def __init__(self, data_name='twitter'):
        # all dataset files for the model

        
        # all the data and corresponding id .
        self.data = data_name + '/all_cascade.txt'
        self.data_id = data_name + '/all_cascade_id.txt'

        self.train_data = data_name + '/cascade.txt'                # train file path.
        self.valid_data = data_name + '/cascadevalid.txt'           # valid file path.
        self.test_data = data_name + '/cascadetest.txt'             # test file path.

        # the cascade id files (id )
        self.train_data_id = data_name + '/cascade_id.txt'          # train id file path.
        self.valid_data_id = data_name + '/cascadevalid_id.txt'     # valid id file path.
        self.test_data_id = data_name + '/cascadetest_id.txt'       # test id file path.

        # user dict and list
        self.u2idx_dict = data_name + '/u2idx.pickle'
        self.idx2u_dict = data_name + '/idx2u.pickle'

        # user and item dict and list
        self.ui2idx_dict = data_name + '/ui2idx.pickle'
        self.idx2ui_dict = data_name + '/idx2ui.pickle'
        
        # social network file
        self.net_data = data_name + '/edges.txt'

        # Bipartite edge file
        self.item_net_data = data_name + "/edges_item.txt"
        
        # save path.
        self.save_path = ''

class Logger(object):
    def __init__(self, Lab_Dir="", dataset="", filename="Default.log"):
        import os
        import sys
        dir_path = f"./log/{Lab_Dir}/{dataset}/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.terminal = sys.stdout

        self.log = open(dir_path+filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def GetPastime(data_name):
        options = Options(data_name)
        max_time = 0
        min_time = 1000000000000
        # max_time_diff = -1
        with open(options.train_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines :
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        with open(options.valid_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines:
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        with open(options.test_data) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines:
                times = [item.split(",")[1] for item in line.split()]
                for time in times:
                    max_len = 10
                    if data_name.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        return max_time - min_time

def GetPastime_twitter(data_path):
        max_time = 0
        min_time = 1000000000000
        # max_time_diff = -1
        with open(data_path) as file:
            lines = [line.split("\n")[0].strip() for line in file.readlines()]
            for line in lines :
                times = [item.split(" ")[1] for item in line.split(",")]
                for time in times:
                    max_len = 10
                    if data_path.find("memetracker") != -1: max_len = 12
                    if len(time) != max_len:
                        int_time = int(time.ljust(max_len, '0'))
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time
                    else:
                        int_time = int(time)
                        if int_time > max_time:
                            max_time = int_time
                        if int_time < min_time:
                            min_time = int_time

        return max_time - min_time


def ConfigFromParser(desc=None, default_task='translation'):

    data_path = "./data/"

    parser = argparse.ArgumentParser(add_help=False, 
                                     allow_abbrev=False,
                                     description='Simple parser.')

    parser.add_argument("--data", type=str, default="android",
                        help='"android", "christianity", "twitter", "memetracker"')
    parser.add_argument('--data_process', type=str, default="DataConstructor", 
                        help='The data processor of the model')
    parser.add_argument('--max_len', type=int, default=200,
                        help='The max length for the model, As information is a sequential task, we should strictly control the max length to be predicted.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='The split ratio for training our model.')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='The split ratio for validating our model.')
    parser.add_argument('--graph_type', default="social+item",
                        choices=['social', 'item', 'social+item'],
                        help='set the edges in the heterogeneous graph type.')
    parser.add_argument('--dynamic_graph_type', default="social+item",
                        choices=['social', 'item', 'social+item'],
                        help='set the edges in the heterogeneous graph type.')
    parser.add_argument('--graph_preprocess', type=lambda x:bool(distutils.util.strtobool(x)), default="False", help='whether to pre-process the dataset with social network')

    parser.add_argument('--model', type=str, default="CasHyper", help='run which type of model')
    parser.add_argument('--time_encoder_type', type=str, default="Interval", help='The time encoder of the model')  # time Encoder
    parser.add_argument('--decoder_type', type=str, default="TransformerBlock", help='The decoder of the model')

    parser.add_argument("--with_st_graph", type=lambda x:bool(distutils.util.strtobool(x)), default="True", help='whether to use the static  graph encoder for the model')
    parser.add_argument("--with_normal_gnn", type=lambda x:bool(distutils.util.strtobool(x)), default="True", help='Set to TRUE to use static graph encoder, when ablation, set this to FALSE and with_dy_grap to True.')

    parser.add_argument("--with_dy_graph", type=lambda x:bool(distutils.util.strtobool(x)), default="True", help='--------whether to use the dynamic graph encoder for the model')
    parser.add_argument("--with_edge_recomputing", type=lambda x:bool(distutils.util.strtobool(x)), default="True", help='------whether to recompute the edge weight to maintain edge dynamics.')
    parser.add_argument("--with_dy_normal_edge_gnn", type=lambda x:bool(distutils.util.strtobool(x)), default="False", help='whether to replace the ODE GNN encoder with the GNN encoder with dynamic edges.')
    parser.add_argument("--dy_merger", type=lambda x:bool(distutils.util.strtobool(x)), default="True", help='whether to use the user embedding merger for the dynamic graph encoder')

    parser.add_argument('--d_model', type=int, default=64, metavar='inputF',
                        help='dimension of initial features.')
    parser.add_argument('--hyper_dim', type=int, default=4, metavar='hyper_dim',
                        help='number of parts.')
    parser.add_argument('--time_step', type=int, default=3,
                        help='number of time steps.')
    parser.add_argument('--graph_hidden_dim', type=int,
                        default=64, help='dimension of the graph encoders.')
    parser.add_argument('--uncertainty_hidden_dim', type=int,
                        default=64, help='dimension of the uncertainty user adapter.')
    parser.add_argument('--time_dim', type=int, default=8,
                        metavar='time', help='The dim of the time encoder')
    parser.add_argument('--pos_dim', type=int, default=8,
                        metavar='pos', help='The dim of the positional embedding')
    parser.add_argument('--heads', type=int, default=1,
                        help='number of heads in transformer')
    parser.add_argument('--time_interval', type=int, default=10000,
                        help='the time interval for each time slice')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16,
                        metavar='BS', help='batch size')
    parser.add_argument('--dropout_rate', type=float,
                        default=0.15, metavar='dropout', help='dropout rate')
    parser.add_argument('--warmup', type=int, default=10)  # warmup epochs
    parser.add_argument('--n_warmup_steps', type=int, default=1000, metavar='LR',
                        help='the warmup steps in the model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--save_path', default=None)
    parser.add_argument('--save_mode', type=str,
                        choices=['all', 'best'], default='best')
    parser.add_argument('--lab_no', default="-1_-1",
                        help='lets take down the number of lab to distinggush the model.')
    parser.add_argument('--notes', default="Testing",
                        help='lets take some notes to distinggush the model.')
    parser.add_argument('--sweep', type=lambda x:bool(distutils.util.strtobool(x)), default="False",
                        help='let sweep do the logging.')
    parser.add_argument('--export_log', type=lambda x:bool(distutils.util.strtobool(x)), default="False",
                        help='copy the log to a certain file.')
    opt = parser.parse_args()

    opt.dataset = opt.data
    opt.data_path = data_path + opt.data
    opt.static_graph_type = opt.graph_type
    
    opt.d_word_vec = opt.d_model
    opt.transformer_dim = opt.d_model + opt.time_dim + opt.pos_dim
    
    opt.lab_no = f"Lab_{opt.lab_no}"
    opt.notes = f"{opt.lab_no}_{opt.notes}"

    if opt.data == "twitter":
        opt.pass_time = GetPastime_twitter(opt.data_path + "/" + "cascades.txt")
    else:
        opt.pass_time = GetPastime(opt.data_path)
    # assert opt.dy_ODE_time_step_split >= 2 
    assert opt.time_step >= 2
    
    opt.seed = SEED
    SeedEverything(SEED)

    if opt.export_log:
        print(
            f"export training details in file: \n \t ./log/{opt.lab_no}/{opt.data}/logfile_{opt.model}_{opt.notes}_{CURRENT_TIME}.log")
        sys.stdout = Logger(Lab_Dir=opt.lab_no, dataset=opt.data,
                            filename=f"logfile_{opt.model}_{opt.notes}_{CURRENT_TIME}.log")

    if not os.path.exists(f"./checkpoints/"): os.makedirs(f"./checkpoints/")
    if not os.path.exists(f"./checkpoints/{opt.data}"): os.makedirs(f"./checkpoints/{opt.data}/")
    opt.save_path = f"./checkpoints/{opt.data}/{opt.model}_{opt.data}_{opt.notes}_{CURRENT_TIME}.pt"
        
    print("*" * 30)
    in_opt_dict = vars(opt)
    for key in in_opt_dict.keys():
        print(f"{key}={in_opt_dict[key]}")
    print("*" * 30)
    
    
    
    return opt


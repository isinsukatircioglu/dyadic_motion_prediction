import easydict
import os
import sys
from log import save_options, save_ckpt, save_csv_log
from pprint import pprint

class Options:
    def __init__(self):
        self.opt = None

    def _initial(self):
        self.opt = easydict.EasyDict({'exp': 'test',
                                      'is_eval': False,
                                      'ckpt': './checkpoint/',
                                      'skip_rate': 1,
                                      'skip_rate_test': 1,
                                      'in_features': 57,  # 66,
                                      'num_stage': 12,
                                      'd_model': 256,
                                      'kernel_size': 10,  # M
                                      'drop_out': 0.5,
                                      'input_n': 60,
                                      'output_n': 30,  # T
                                      'dct_n': 40,  # M+T
                                      'itera': 1,
                                      'lr_now': 0.0005,
                                      'max_norm': 10000,
                                      'epoch': 500,
                                      'batch_size': 32,
                                      'test_batch_size': 32,
                                      'is_load': True #False for a new training
                                      })

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        if not self.opt.is_eval:
            script_name = os.path.basename(sys.argv[0])[:-3]
            log_name = 'Dyadic_{}_in{}_out{}_ks{}_dctn{}_stage{}_d{}_bs{}'.format(script_name, self.opt.input_n,
                                                                            self.opt.output_n,
                                                                            self.opt.kernel_size,
                                                                            self.opt.dct_n,
                                                                            self.opt.num_stage,
                                                                            self.opt.d_model,
                                                                            self.opt.batch_size)
            self.opt.exp = log_name
            # do some pre-check
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                save_options(self.opt)
            self.opt.ckpt = ckpt
            save_options(self.opt)
        self._print()
        return self.opt

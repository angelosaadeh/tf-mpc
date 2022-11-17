import main
import numpy as np
import time
import sys
sys.path.append("..")
import mpc.data.data as data
import mpc.mpc as mpc


if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    v = data.Unpack().get_array('data/mnist.alice')

    main.function(prot, v)
    
    time.sleep(3)

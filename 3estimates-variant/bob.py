import main
import numpy as np
import sys
sys.path.append("..")
import mpc.data.data as data
import mpc.mpc as mpc


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 20, 2**32)

    v = data.Unpack().get_array('data/hubdub.bob')
    v = v.transpose()
    main.function(prot, v)

    exit()

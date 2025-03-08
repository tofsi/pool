from environment import PoolEnvironment
import sys
from pathlib import Path
import numpy as np
from poolsim.pool import config
from play_pool import simulate

spinningup_path = str(Path.cwd())+"/spinningup_tf2"
print(spinningup_path)
sys.path.append(spinningup_path)
import spinningup_tf2.spinup_bis as spinup

if __name__ == "__main__":
    state = np.array([[500,
                        200], [600,200]], dtype=np.float32)
    pocketed = np.zeros(2, np.int8)
    action = np.array([1.0, 1.056])
    print(simulate(state, pocketed, action))
    # def environment_function(): return PoolEnvironment()
    #spinup.ddpg_tf2(environment_function)
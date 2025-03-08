from environment import PoolEnvironment
import sys
sys.path.append('/home/hedwig/Documents/Python/spinningup_tf2')
import spinup_bis as spinup
if __name__ == "__main__":
    def environment_function(): return PoolEnvironment()
    spinup.ddpg_tf2(environment_function)
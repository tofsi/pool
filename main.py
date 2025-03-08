from environment import PoolEnvironment
import sys
from pathlib import Path
spinningup_path = str(Path.cwd())+"/pool/spinningup_tf2"
sys.path.append(spinningup_path)
import spinningup_tf2.spinup_bis as spinup
if __name__ == "__main__":
    def environment_function(): return PoolEnvironment()
    spinup.ddpg_tf2(environment_function)
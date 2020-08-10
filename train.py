from model_mpe import maac_mpe
import argparse
from env_list import mpe_list
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scenario",
        help="set the scenario.",
        type=str
    )
    args = parser.parse_args()
    env_id = args.scenario
    if env_id not in mpe_list:
        print('Don\'t support the env id!')
        sys.exit(0)
    os.makedirs('./models/{}'.format(env_id), exist_ok=True)
    # * the size of replay buffer must be appropriate
    test = maac_mpe(
        env_id=env_id,
        batch_size=1024,
        learning_rate=1e-3,
        exploration=0,
        episode=50000,
        gamma=0.99,
        alpha=0.01,
        capacity=1000000,
        rho=0.999,
        update_iter=4,
        update_every=100,
        head_dim=32,
        traj_len=25,
        render=False
    )
    test.run()
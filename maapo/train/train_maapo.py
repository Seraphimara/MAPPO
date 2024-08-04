from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv
from ray import tune
from maapo.algo_maapo.maapo import MAAPO
from maapo.utils.callbacks import MultiAgentDrivingCallbacks
from maapo.utils.env_wrappers import create_gym_wrapper, get_rllib_compatible_env, get_ccppo_env
from maapo.utils.train import train
from maapo.utils.utils import get_train_parser

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    stop = int(1200000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=tune.grid_search(
            [
                # get_rllib_compatible_env(MultiAgentParkingLotEnv),
                get_ccppo_env(create_gym_wrapper(MultiAgentIntersectionEnv)),
                # get_rllib_compatible_env(MultiAgentTollgateEnv),
                #

                # get_rllib_compatible_env(MultiAgentRoundaboutEnv),
                # get_rllib_compatible_env(MultiAgentMetaDrive),
                # get_ccppo_env(create_gym_wrapper(MultiAgentRoundaboutEnv)),
                # get_ccppo_env(MultiAgentIntersectionEnv)
            ]
        ),
        #==env_config=dict(start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]), ),
        env_config=dict(num_agents=20, map_config=dict(exit_length=60, lane_num=3)),
        # env_config=dict(num_agents=20, map_config=dict(exit_length=60, lane_num=3),
        #                 # start_seed=tune.grid_search([5000, 6000, 7000, 8000, ]),
        #                 # use_render=True
        #                 ),

        # ===== Resource =====
        # So we need 2 CPUs per trial, 0.25 GPU per trial!
        num_gpus=0.5, #for local worker
        # spf_lr = tune.grid_search([1e-5, 6e-5]),
        # ppo_lr = tune.grid_search([1e-5, 1e-4]),
        num_workers=10, # besides localworker, num of rollout env samplers
        # vf_clip_param=tune.grid_search([80])
        # num_gpus_per_worker = 0.1,
        num_cpus_for_local_worker=5,
        num_cpus_per_worker=0.5,
        # batch_mode = "complete_episodes"
    )

    # Launch training
    train(
        MAAPO,
        exp_name=exp_name,
        keep_checkpoints_num=10,
        stop=stop,
        config=config,
        num_gpus=1,
        num_seeds=1,
        # test_mode=args.test,
        custom_callback=MultiAgentDrivingCallbacks,
        checkpoint_freq=50,
        # fail_fast='raise',
        local_mode=False
    )

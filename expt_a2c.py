import pandas as pd
import custom_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import gym


if __name__ == "__main__":

    df = pd.read_csv('data/STOCKS_GOOGL.csv')
    print(df.head())
    print('df:', df.shape)

    window_size = 10
    start_index = window_size
    end_index = len(df)

    print('window_size:', window_size,
          'start_index:', start_index,
          'end_index:', end_index
          )

    # vectorize the env
    env = DummyVecEnv([lambda: custom_env.StocksEnv(df, window_size, (start_index, end_index))])

    # build A2C
    model = A2C('MlpPolicy', env, verbose=1)

    # learn A2C
    model.learn(total_timesteps=1000)

    # test env
    env_maker = lambda: gym.make(
        'stocks-v0',
        df=df,
        window_size=window_size,
        frame_bound=(start_index, end_index)
    )
    env = DummyVecEnv([lambda: custom_env.StocksEnv(df, window_size, (start_index, end_index))])
    observation = env.reset()


    while True:
        # observation = observation[np.newaxis, ...]
        observation = observation.reshape(observation.shape[-2], observation.shape[-1])

        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step([action])

        # env.render()
        if done:
            print("info:", info)
            break














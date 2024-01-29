import numpy as np
from model import SSSDS4

train = np.load("datasets/mujoco/train.npy")
test = np.load("datasets/mujoco/test.npy")

train_obs_mask = np.random.uniform(0, 1, size = train.shape)
train_obs_mask = train_obs_mask >= 0.1
train = train * train_obs_mask

test_obs_mask = np.random.uniform(0, 1, size = test.shape)
test_obs_mask = test_obs_mask >= 0.1
test = test * test_obs_mask

train_config={
    "output_directory": "./results/mujoco",
    "epochs": 200,
    "epochs_per_ckpt": 50,
    "val_size": 0.01,
    "learning_rate": 2e-4,
    "only_generate_missing": 1,
    "missing_r": "rand",
    "batch_size": 8,
    "verbose": 1
}

model = SSSDS4(*train.shape[1:])
model.set_train_config(**train_config)

model.train(train, train_obs_mask)
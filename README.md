# PPO implementation

My implementation of Proximal Policy Optimization (PPO) adapted from Haozhi Qi's and Eric Yang Yu's implementations. Original PPO paper: https://arxiv.org/abs/1707.06347

## Setup

Please install libraries in the `requirements.txt` file using the following command:

```bash
pip install -r requirements.txt
```

## Use
Create a virtual environment and install requirements.
```
python -m venv ~/envs/PPO
source ~/envs/PPO/bin/activate
pip install -r requirements.txt
```

To train an agent from scratch:
```
python main.py --test 0 --experiment <experiment_name>
```

To visualize an agents performance:
```
python main.py --test 1 --checkpoint <checkpoint_name>
```

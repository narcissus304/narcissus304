import os
import subprocess

# 修改这些路径为你的配置路径
data_config_path = 'path_to_data_config.yaml'
model_config_path = 'path_to_model_config.cfg'
weights_path = 'path_to_pretrained_weights.pt'
batch_size = 16

command = [
    'python', 'train.py',
    '--data', data_config_path,
    '--cfg', model_config_path,
    '--weights', weights_path,
    '--batch-size', str(batch_size)
]

# 运行训练命令
result = subprocess.run(command, capture_output=True, text=True)

# 打印训练日志
print(result.stdout)
print(result.stderr)
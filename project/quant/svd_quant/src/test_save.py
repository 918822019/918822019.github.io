import torch
import torch.nn as nn

# 创建一个简单的测试
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(10, 10))
    
    def forward(self, x):
        return x @ self.param

# 创建模块
module = TestModule()

# 保存权重
weights = {'param': module.param.data.cpu()}
torch.save(weights, 'output/test_weights.pt')
print('Weights saved successfully')

# 检查文件是否存在
import os
if os.path.exists('output/test_weights.pt'):
    print('File exists')
else:
    print('File does not exist')

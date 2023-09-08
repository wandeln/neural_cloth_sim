from setups import Dataset
import matplotlib.pyplot as plt
from cloth_net import Cloth_net
from get_param import params

dataset = Dataset(params.height,params.width)

cloth_net = Cloth_net(params.SMP_model_type,params.SMP_encoder_name)

x_v,_ = dataset.ask()

a = cloth_net(x_v)

print(f"a.shape: {a.shape}")

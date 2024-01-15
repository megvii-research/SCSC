from opCounter import *
from cbtnet import cbtnet50 as CBTNet
from opCounter import profile
from opCounter.utils import clever_format


net = CBTNet()
flops, params = profile(net, input_size=(1,3,224,224))
print('flops = {} params = {}'.format(flops, params))
print('flops = {} params = {}'.format(clever_format(flops), clever_format(params)))


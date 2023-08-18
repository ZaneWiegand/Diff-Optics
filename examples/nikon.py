# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.append("../")
import diffoptics as do
from datetime import datetime

"""
DISCLAIMER:

This script was used to generate Figure 11 in the paper. However, the results produced in
the paper mistakenly assumed the air refractive index to be 1, rather than 1.000293. This 
slight difference will produce a slightly different result from the paper.

If you want to reproduce the result in the paper, change the following item in

diffoptics.basics.Material.__init__.MATERIAL_TABLE

from:

"air":        [1.000293, np.inf],

to

"air":        [1.000000, np.inf],

And re-run this script.
"""

# %%
# initialize a lens
device = torch.device('cpu')
# device = torch.device('cuda')
lens = do.Lensgroup(device=device)
# %%
# load optics
lens.load_file(Path('./lenses/Zemax_samples/Nikon-z35-f1.8-JPA2019090949-example2.txt'))
lens.plot_setup2D()
# %%
# sample wavelengths in [nm]
wavelengths = torch.Tensor([656.2725, 587.5618, 486.1327]).to(device)
views = np.array([0, 10, 20, 32.45])
colors_list = 'bgry'
ax, fig = lens.plot_setup2D_with_trace(views,wavelengths[1], M=5,entrance_pupil=True)
ax.axis('off')
ax.set_title("")
fig
# fig.savefig("../results/nikon/layout_trace_" + "_" + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + ".pdf", bbox_inches='tight')
# %%
def plot_layout(string):
    ax, fig = lens.plot_setup2D_with_trace(views, wavelengths[1], M=5, entrance_pupil=True)
    ax.axis('off')
    ax.set_title("")
    fig.savefig("../results/nikon/layout_trace_" + string + "_" + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + ".pdf", bbox_inches='tight')

M = 31
def render(verbose=False, entrance_pupil=False):
    def render_single(wavelength):
        pss = []
        spot_rms = []
        loss = 0.0
        for view in views:
            ray = lens.sample_ray(wavelength, view=view, M=M, sampling='grid', entrance_pupil=entrance_pupil)
            ps = lens.trace_to_sensor(ray, ignore_invalid=True)

            # calculate RMS
            tmp, ps = lens.rms(ps[...,:2], squared=True)
            loss = loss + tmp #! 各个视场的 rms 的平方相加作为 loss
            pss.append(ps) #! 记录各个视场的像面上的点坐标
            spot_rms.append(np.sqrt(tmp.item()))
        return pss, loss, np.array(spot_rms)

    pss_all = []
    rms_all = []
    loss = 0.0
    for wavelength in wavelengths:
        if verbose:
            print("Rendering wavelength = {} [nm] ...".format(wavelength.item()))
        #! pss 单波长下各个视场的点坐标，loss_single 单波长下各个视场的 rms 的平方和，rmss 为各个视场的 rms
        pss, loss_single, rmss = render_single(wavelength)
        loss = loss + loss_single
        pss_all.append(pss)
        rms_all.append(rmss)
    return pss_all, loss, np.array(rms_all)

def func():
    ps = render()[0]
    return torch.vstack([torch.vstack(ps[i]) for i in range(len(ps))])

def loss_func():
    return render()[1]

def info(string):
    loss, rms = render()[1:]
    print("=== {} ===".format(string))
    print("loss = {}".format(loss))
    print("==========")
    plot_layout(string)
    return rms #! 各个波长各个视场下的 rms
# %%
rms_org = info('original')
print(rms_org.mean())
# %%
id_range = list(range(0, 19)) #! 最后三个面是平板玻璃前后表面和像面，因此不在优化范围内，第一面物面也不在优化范围内
id_range.pop(lens.aperture_ind) #! POP 出孔阑的序号
id_asphere = [16, 17] #! 这两面是非球面
for i in id_asphere:
    lens.surfaces[i].ai = torch.Tensor([0.0]).to(device) #! 把这两个非球面的非球面系数设置成 0 
# %%
diff_names = []
diff_names += ['surfaces[{}].c'.format(str(i)) for i in id_range] #! 只优化各个面的曲率
diff_names += ['surfaces[{}].k'.format(str(i))  for i in id_asphere] #! 优化非球面的 k 系数
diff_names += ['surfaces[{}].ai'.format(str(i)) for i in id_asphere] #! 优化非球面的各个高阶项系数
# %%
rms_init = info('initial')
print(rms_init.mean())
# %%
# optimize
start = torch.cuda.Event(enable_timing=True) #! 记录时间开始
end = torch.cuda.Event(enable_timing=True) #! 记录时间结束

start.record()
#TODO do.LM 的优化具体过程？？？
out = do.LM(lens, diff_names, 1e-2, option='diag').optimize(func, lambda y: 0.0 - y, maxit=100, record=True)
end.record()
torch.cuda.synchronize()
print('Finished in {:.2f} mins'.format(start.elapsed_time(end)/1000/60))
# %%
rms_opt = info('optimized')
print(rms_opt.mean())
# %%
# plot loss
fig, ax = plt.subplots(figsize=(12,6))
ax.semilogy(out['ls'], 'k-o', linewidth=3)
plt.xlabel('iteration')
plt.ylabel('error function')
plt.savefig("../results/nikon/ls_nikon.pdf", bbox_inches='tight')
# %%
def save_fig(xs, string):
    fig, ax = plt.subplots(figsize=(3,1.5))
    xs = xs.T
    for i, x in enumerate(xs):
        ax.semilogy(x, colors_list[i], marker='o', linewidth=1)
        plt.xlabel('wavelength [nm]')
        plt.ylabel('RMS spot size [um]')
    plt.ylim([0.08, 50])
    plt.xticks([0,1,2], ['656.27', '587.56', '486.13'])
    plt.yticks([0.1,1,10,50], ['0.1', '1', '10', '50'])
    fig.savefig("../results/nikon/rms_" + string + "_nikon.pdf", bbox_inches='tight')
# %%
save_fig(rms_init * 1e3, "init")
save_fig(rms_org * 1e3, "org")
save_fig(rms_opt * 1e3, "opt")
# %%

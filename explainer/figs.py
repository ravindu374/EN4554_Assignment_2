"""Build every data figure for the explainer PDF.

Only reads cached results/*.json and the CIFAR images. Trains nothing.
"""
import json, os, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, 'results')
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.bbox': 'tight',
})
BLUE, RED, GREEN, GREY = '#2563eb', '#dc2626', '#059669', '#9ca3af'

def load(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.svg'), format='svg')
    plt.close(fig)
    print('  wrote', name + '.svg')


# ---------------------------------------------------------------- Task 1(c)
t1c = load('t1c_mlp_sgd')
if t1c:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].plot(t1c['epoch'], t1c['train_loss'], color=BLUE)
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('training loss')
    ax[0].set_title('Loss falls steadily')
    ax[0].axhline(np.log(10), ls=':', color=GREY)
    ax[0].annotate('ln 10 = 2.303\n(knowing nothing)', xy=(2, np.log(10)),
                   xytext=(6, 1.95), fontsize=7.5, color='#555')
    ax[0].grid(alpha=.25)

    ax[1].plot(t1c['epoch'], [100*a for a in t1c['train_acc']], color=BLUE, label='train')
    ax[1].plot(t1c['epoch'], [100*a for a in t1c['val_acc']], color=RED, label='validation')
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('accuracy (%)')
    ax[1].set_title('...but validation stops at ~50%')
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.25)
    gap = 100*(t1c['train_acc'][-1] - t1c['val_acc'][-1])
    ax[1].annotate('', xy=(30, 100*t1c['train_acc'][-1]), xytext=(30, 100*t1c['val_acc'][-1]),
                   arrowprops=dict(arrowstyle='<->', color='#444', lw=1.1))
    ax[1].text(27.5, 70, 'gap\n%.0f pts' % gap, fontsize=7.5, ha='right', color='#444')
    save(fig, 'f_t1c')


# ---------------------------------------------------------------- Task 1(d)
t1d = load('t1d_grad_norms')
if t1d:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    L = range(1, t1d['depth']+1)
    ax.semilogy(L, t1d['sigmoid'], 'o-', color=RED,  ms=4, label='sigmoid + Xavier')
    ax.semilogy(L, t1d['relu'],    's-', color=BLUE, ms=4, label='ReLU + Kaiming')
    ax.set_xlabel('layer  (1 = input side,  10 = output side)')
    ax.set_ylabel(r'gradient norm $\|\partial L/\partial W\|$')
    ax.set_title('Sigmoid gradients die on the way back; ReLU gradients do not')
    ax.set_xticks(list(L)); ax.grid(alpha=.25, which='both'); ax.legend(fontsize=8)
    r = t1d['sigmoid'][-1]/t1d['sigmoid'][0]
    ax.annotate('%.0f x smaller\nat layer 1' % r, xy=(1, t1d['sigmoid'][0]),
                xytext=(2.4, 3e-5), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1))
    save(fig, 'f_t1d')


# ---------------------------------------------------- Task 2(b) trajectories
def f(x, y):  return 0.5*(x**2 + 50*y**2)
def grad(p):  return np.array([p[0], 50*p[1]])

def run_sgd(p0, lr, steps, mu=0.0):
    p, b, path = np.array(p0, float), np.zeros(2), [np.array(p0, float)]
    for _ in range(steps):
        g = grad(p)
        if mu: b = mu*b + g; step = b
        else:  step = g
        p = p - lr*step; path.append(p.copy())
    return np.array(path)

def run_adam(p0, lr, steps, b1=.9, b2=.999, eps=1e-8):
    p, m, v, path = np.array(p0, float), np.zeros(2), np.zeros(2), [np.array(p0, float)]
    for t in range(1, steps+1):
        g = grad(p)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g*g
        p = p - lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        path.append(p.copy())
    return np.array(path)

P = {'SGD  (lr 0.038)':            (run_sgd((-9.,1.), .038, 60),        RED),
     'SGD + momentum  (lr 0.038)': (run_sgd((-9.,1.), .038, 60, .9),    BLUE),
     'Adam  (lr 0.3)':             (run_adam((-9.,1.), .3, 60),         GREEN)}

fig, ax = plt.subplots(figsize=(7.2, 2.9))
xs, ys = np.linspace(-10, 10, 300), np.linspace(-1.4, 1.4, 300)
XX, YY = np.meshgrid(xs, ys)
cs = ax.contour(XX, YY, f(XX, YY), levels=[.5,2,5,10,20,40,65,100], colors=GREY, linewidths=.6)
ax.clabel(cs, inline=True, fontsize=6, fmt='%g')
for lab, (pa, col) in P.items():
    flips = int(np.sum(np.diff(np.sign(pa[:,1])) != 0))
    ax.plot(pa[:,0], pa[:,1], 'o-', color=col, ms=2, lw=1.1,
            label='%s  -  %d zigzags' % (lab, flips))
ax.plot(0, 0, 'k*', ms=12); ax.plot(-9, 1, 'ks', ms=5)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(r'Narrow valley: $f(x,y)=\frac{1}{2}(x^2+50y^2)$   -   start at $(-9,1)$, 60 steps')
ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=.2)
save(fig, 'f_t2b')


# ---------------------------------------------------------------- Task 2(c)
t2c = load('t2c_lr_sweep')
if t2c:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    cols = {'SGD': RED, 'SGD+momentum': BLUE, 'Adam': GREEN}
    for o in t2c['opts']:
        accs = [100*t2c['runs']['%s@%g' % (o, lr)]['best_val_acc'] for lr in t2c['lrs']]
        ax.semilogx(t2c['lrs'], accs, 'o-', color=cols[o], ms=6, label=o)
        for lr, a in zip(t2c['lrs'], accs):
            ax.annotate('%.0f' % a, (lr, a), textcoords='offset points',
                        xytext=(0, 6), fontsize=7, ha='center', color=cols[o])
    ax.set_xlabel('learning rate'); ax.set_ylabel('best validation accuracy (%)')
    ax.set_title('How much does the learning rate matter?')
    ax.legend(fontsize=8, loc='lower center'); ax.grid(alpha=.25, which='both')
    save(fig, 'f_t2c')


# ---------------------------------------------------------------- Task 3(a)
t3a = load('t3a_regularization')
if t3a:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    names = t3a['order']
    tr  = [100*t3a['runs'][k]['final_train_acc'] for k in names]
    va  = [100*t3a['runs'][k]['best_val_acc']    for k in names]
    gp  = [100*t3a['runs'][k]['gap']             for k in names]
    y = np.arange(len(names))
    ax[0].barh(y-0.2, tr, .4, color='#93c5fd', label='train acc')
    ax[0].barh(y+0.2, va, .4, color=BLUE,      label='best val acc')
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel('%'); ax[0].set_title('Train vs validation'); ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=.25, axis='x')

    colors = [RED if g > 25 else ('#f59e0b' if g > 12 else GREEN) for g in gp]
    ax[1].barh(y, gp, .55, color=colors)
    for i, g in enumerate(gp):
        ax[1].text(g+0.6, i, '%.1f' % g, va='center', fontsize=7.5)
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel('train - val  (percentage points)')
    ax[1].set_title('The memorisation gap (smaller is better)')
    ax[1].grid(alpha=.25, axis='x')
    save(fig, 'f_t3a')


# ---------------------------------------------------------------- Task 3(b)
t3b = load('t3b_depth')
if t3b:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    shades = {8: '#93c5fd', 20: '#3b82f6', 56: '#b91c1c'}
    for i, fam in enumerate(('plain', 'residual')):
        for d in t3b['depths']:
            h = t3b['runs']['%s-%d' % (fam, d)]
            ax[i].semilogy(range(1, len(h['train_loss'])+1), h['train_loss'],
                           color=shades[d], lw=1.8, label='%s-%d' % (fam, d))
        ax[i].set_xlabel('epoch'); ax[i].grid(alpha=.25, which='both')
        ax[i].legend(fontsize=8)
    ax[0].set_ylabel('training loss (log scale)')
    ax[0].set_title('Plain: 56 layers trains WORSE than 20')
    ax[1].set_title('Residual: deeper keeps helping')
    p56 = t3b['runs']['plain-56']['final_train_loss']
    r56 = t3b['runs']['residual-56']['final_train_loss']
    ax[0].annotate('stuck at %.2f' % p56, xy=(28, p56), xytext=(12, 0.25),
                   fontsize=8, color='#b91c1c',
                   arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=1))
    save(fig, 'f_t3b')

    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    d = t3b['depths']
    pl = [t3b['runs']['plain-%d' % x]['final_train_loss'] for x in d]
    rs = [t3b['runs']['residual-%d' % x]['final_train_loss'] for x in d]
    ax.semilogy(d, pl, 'o-', color=RED,  ms=7, label='plain')
    ax.semilogy(d, rs, 's-', color=BLUE, ms=7, label='residual')
    for x, v in zip(d, pl): ax.annotate('%.4f' % v, (x, v), xytext=(0,-14),
                                        textcoords='offset points', fontsize=7.5, ha='center', color=RED)
    for x, v in zip(d, rs): ax.annotate('%.4f' % v, (x, v), xytext=(0,7),
                                        textcoords='offset points', fontsize=7.5, ha='center', color=BLUE)
    ax.set_xticks(d); ax.set_xlabel('depth (layers)'); ax.set_ylabel('final training loss')
    ax.set_title('More layers should never hurt - but for plain nets it does')
    ax.legend(fontsize=8); ax.grid(alpha=.25, which='both')
    save(fig, 'f_t3b_summary')


# ------------------------------------------------- augmentation illustration
try:
    from torchvision import datasets
    ds = datasets.CIFAR10(root=os.path.join(ROOT, 'data'), train=True, download=False)
    rng = np.random.default_rng(3)
    idx = [4, 7, 21]
    imgs = [ds.data[i].astype(np.float32)/255. for i in idx]
    names = [ds.classes[ds.targets[i]] for i in idx]

    fig, ax = plt.subplots(1, 5, figsize=(7.4, 1.85))
    ax[0].imshow(imgs[0]); ax[0].set_title('original\n(%s)' % names[0], fontsize=8)

    flipped = imgs[0][:, ::-1, :]
    ax[1].imshow(flipped); ax[1].set_title('flip\n(still a %s)' % names[0], fontsize=8)

    pad = np.pad(imgs[0], ((4,4),(4,4),(0,0)))
    oy, ox = 7, 1
    ax[2].imshow(pad[oy:oy+32, ox:ox+32]); ax[2].set_title('random crop\n(shifted)', fontsize=8)

    cut = imgs[0].copy(); cut[12:20, 6:14] = imgs[0].mean(axis=(0,1))
    ax[3].imshow(cut); ax[3].set_title('cutout\n(hide a patch)', fontsize=8)

    lam = 0.6
    mixed = lam*imgs[0] + (1-lam)*imgs[1]
    ax[4].imshow(np.clip(mixed, 0, 1))
    ax[4].set_title('mixup  $\\lambda$=0.6\n60%% %s / 40%% %s' % (names[0], names[1]), fontsize=7.5)

    for a in ax: a.set_xticks([]); a.set_yticks([])
    save(fig, 'f_augment')
except Exception as e:
    print('  skipped augmentation figure:', e)

print('done -> figs/')

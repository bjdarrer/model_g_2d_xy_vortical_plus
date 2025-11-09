"""
Model G (2D proper, x–y) with Vortical Motion — PLUS vorticity panel, quiver overlay,
fluid parameters on CLI, and rotating-seed preset. SAFE + RESUMABLE.

- Written by Brendan Darrer aided by ChatGPT5 date: 8th November 2025
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT5 writing it and Brendan guiding it to produce a clean code.

- Periodic pseudo-spectral solver (FFT) for spatial ops; semi-Lagrangian advection.
- Coupled compressible flow u=(ux,uy) with isothermal pressure and viscosity.
- 3D surface of Y, G, X/10; optional 2D vorticity panel and quiver overlay.

Install:
  pip install numpy scipy matplotlib imageio imageio[ffmpeg]

Batch:
  python3 model_g_2d_xy_vortical_plus__2b.py --nx 192 --ny 192 --Lx 60 --Ly 60 \
      --Tfinal 8 --dt 0.005 --segment_dt 0.5 --zlim 1.0 \
      --alphaG 0.02 --alphaX 0.02 --alphaY 0.02 --cs2 1.0 --nu 0.25 \
      --rotseed --swirl_amp 1.0 --swirl_sigma 6.0 --swirl_cx 30 --swirl_cy 30 \
      --vort_panel --quiver --quiver_stride 8

Live (needs interactive backend; e.g., MPLBACKEND=TkAgg):
  MPLBACKEND=TkAgg python3 model_g_2d_xy_vortical_plus__2b.py --live --rotseed --vort_panel --quiver
"""
import os
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description='Model G 2D + vortical motion (spectral, periodic) with viz extras')
# domain/grid/time
parser.add_argument("--Dg", type=float, default=1.0, help="Diffusion coefficient for G")
parser.add_argument("--Dx", type=float, default=1.0, help="Diffusion coefficient for X")
parser.add_argument("--Dy", type=float, default=12.0, help="Diffusion coefficient for Y")
parser.add_argument('--nx', type=int, default=192)
parser.add_argument('--ny', type=int, default=192)
parser.add_argument('--Lx', type=float, default=60.0)
parser.add_argument('--Ly', type=float, default=60.0)
parser.add_argument('--Tfinal', type=float, default=10.0)
parser.add_argument('--segment_dt', type=float, default=0.5)
parser.add_argument('--dt', type=float, default=0.005)
parser.add_argument('--nt_anim', type=int, default=200)
parser.add_argument('--max_frames', type=int, default=200)
# model-G params (eqs17 style kept)
parser.add_argument('--a', type=float, default=14.0)
parser.add_argument('--b', type=float, default=29.0)
parser.add_argument('--dxcoef', type=float, default=1.0)
parser.add_argument('--dycoef', type=float, default=12.0)
parser.add_argument('--pcoef', type=float, default=1.0)
parser.add_argument('--qcoef', type=float, default=1.0)
parser.add_argument('--gcoef', type=float, default=0.1)
parser.add_argument('--scoef', type=float, default=0.0)
parser.add_argument('--ucross', type=float, default=0.0)
# fluid coupling
parser.add_argument('--alphaG', type=float, default=0.02)
parser.add_argument('--alphaX', type=float, default=0.02)
parser.add_argument('--alphaY', type=float, default=0.02)
parser.add_argument('--cs2', type=float, default=1.0)
parser.add_argument('--nu', type=float, default=0.25)
# seeds
parser.add_argument('--Tseed', type=float, default=3.0)
parser.add_argument('--seed_sigma_space', type=float, default=2.0)
parser.add_argument('--seed_sigma_time', type=float, default=2.0)
parser.add_argument('--seed_center', type=float, nargs=2, default=None, help='(xc yc), default center of box')
# rotating seed for velocity
parser.add_argument('--rotseed', action='store_true', help='Initialize a swirling velocity seed')
parser.add_argument('--swirl_amp', type=float, default=1.0, help='Swirl tangential speed amplitude')
parser.add_argument('--swirl_sigma', type=float, default=6.0, help='Gaussian radius of swirl')
parser.add_argument('--swirl_cx', type=float, default=None, help='Swirl center x (default mid-box)')
parser.add_argument('--swirl_cy', type=float, default=None, help='Swirl center y (default mid-box)')
# viz
parser.add_argument('--zlim', type=float, default=1.0)
parser.add_argument('--downsample', type=int, default=0)
parser.add_argument('--vort_panel', action='store_true', help='Add a 2D vorticity panel beside the 3D surface')
parser.add_argument('--quiver', action='store_true', help='Draw quiver arrows over vorticity panel')
parser.add_argument('--quiver_stride', type=int, default=8)
parser.add_argument('--live', action='store_true', help='Enable live 3D viewer')
parser.add_argument('--live_stride', type=int, default=5)
args = parser.parse_args()

# ---------------- Paths ----------------
run_name  = 'model_g_2d_xy_vortical_plus__2b'
out_dir   = f'out_{run_name}'
frames_dir= os.path.join(out_dir, 'frames')
ckpt_path = os.path.join(out_dir, 'checkpoint_vortical_plus.npz')
mp4_path  = os.path.join(out_dir, f'{run_name}.mp4')
final_png = os.path.join(out_dir, 'final_snapshot.png')
os.makedirs(frames_dir, exist_ok=True)

# --- Diffusion coefficients & timestep -----------------------------
Dg = 1.0
Dx = 1.0
Dy = 12.0
dt = args.dt

# --- Grid and Fourier setup (final stable version) ----------------
Lx, Ly = args.Lx, args.Ly
nx, ny = args.nx, args.ny
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")  # shape (ny, nx)


# Correct Fourier grid shapes: (ny, nx//2+1)
kx = 2 * np.pi * np.fft.rfftfreq(nx, d=Lx/nx)   # → (nx//2+1,)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)    # → (ny,)
KX, KY = np.meshgrid(kx, ky, indexing="xy")      # ✅ not "ij"
K2 = KX**2 + KY**2
K2[0, 0] = 1.0

GammaG = np.exp(-Dg * K2 * dt)
GammaX = np.exp(-Dx * K2 * dt)
GammaY = np.exp(-Dy * K2 * dt)
print("K2:", K2.shape, "GammaG:", GammaG.shape)

"""
# rfft2-compatible wavenumbers → shape (ny, nx//2+1)
kx = 2 * np.pi * np.fft.rfftfreq(nx, d=Lx/nx)   # (nx//2+1,)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)    # (ny,)
KY, KX = np.meshgrid(ky, kx, indexing="ij")     # (ny, nx//2+1)
K2 = KX**2 + KY**2
K2[0, 0] = 1.0  # avoid /0
"""

# --- FFT helpers ---------------------------------------------------
def rfft2c(f):
    return np.fft.rfft2(f, s=(ny, nx))

def irfft2c(F):
    return np.fft.irfft2(F, s=(ny, nx))

def grad(f):
    F = rfft2c(f)
    fx = irfft2c(1j * KX * F)
    fy = irfft2c(1j * KY * F)
    return fx, fy

def laplacian(f):
    F = rfft2c(f)
    return irfft2c(-K2 * F)

"""
# --- Diffusion coefficients & timestep -----------------------------
Dg = 1.0
Dx = 1.0
Dy = 12.0
dt = args.dt

# --- Diffusion propagators (same shape as rfft2 output) ------------
GammaG = np.exp(-Dg * K2 * dt)
GammaX = np.exp(-Dx * K2 * dt)
GammaY = np.exp(-Dy * K2 * dt)
print("K2:", K2.shape, "GammaG:", GammaG.shape)
"""

#Ftest = rfft2c(np.random.rand(ny, nx))
#print("Ftest:", Ftest.shape)

Ftest = rfft2c(np.random.rand(ny, nx))
print("Ftest:", Ftest.shape)
assert Ftest.shape == GammaG.shape, f"GammaG {GammaG.shape} != Ftest {Ftest.shape}"

# ---------------- Parameters ----------------
params = {
    'a': args.a,
    'b': args.b,
    'dx': args.dxcoef,
    'dy': args.dycoef,
    'p': args.pcoef,
    'q': args.qcoef,
    'g': args.gcoef,
    's': args.scoef,
    'u_cross': args.ucross,
}
alphaG, alphaX, alphaY = args.alphaG, args.alphaX, args.alphaY
cs2, nu = args.cs2, args.nu

# Homogeneous state
G0 = (params['a'] + params['g']*0.0) / (params['q'] - params['g']*params['p'])
X0 = (params['p']*params['a'] + params['q']*0.0) / (params['q'] - params['g']*params['p'])
Y0 = ((params['s']*X0**2 + params['b']) * X0 / (X0**2 + params['u_cross'])) if (X0**2 + params['u_cross'])!=0 else 0.0
print(f"Homogeneous state: G0={G0:.6g}, X0={X0:.6g}, Y0={Y0:.6g}")

# ---------------- Initial conditions ----------------
pG = np.zeros((ny, nx)); pX = np.zeros((ny, nx)); pY = np.zeros((ny, nx))
ux = np.zeros((ny, nx)); uy = np.zeros((ny, nx))

# Optional rotating seed in velocity
if args.rotseed:
    cx = args.swirl_cx if args.swirl_cx is not None else Lx/2
    cy = args.swirl_cy if args.swirl_cy is not None else Ly/2
    dxg = X - cx
    dyg = Y - cy
    r2 = dxg*dxg + dyg*dyg
    vtheta = args.swirl_amp * np.exp(-r2 / (2*args.swirl_sigma**2))
    R = np.sqrt(r2) + 1e-12
    # tangential components: (-sinθ, cosθ) = (-y/r, x/r)
    ux = - vtheta * (dyg / R)
    uy =   vtheta * (dxg / R)

# ---------------- Forcing chi(x,y,t) ----------------
def bell(s, x):
    return np.exp(-(x/s)**2 / 2.0)

Tseed = args.Tseed
seed_sigma_space = args.seed_sigma_space
seed_sigma_time  = args.seed_sigma_time
if args.seed_center is None:
    seed_centers = [(Lx/2, Ly/2)]
else:
    seed_centers = [tuple(args.seed_center)]

def chi_xy_t(t):
    spatial = np.zeros((ny, nx))
    for (xc, yc) in seed_centers:
        spatial += np.exp(-(((X-xc))**2 + ((Y-yc))**2) / (2*seed_sigma_space**2))
    return -spatial * bell(seed_sigma_time, t - Tseed)

# ---------------- Utilities ----------------
def auto_downsample(nx, ny, user_ds):
    if user_ds and user_ds > 0:
        return max(1, int(user_ds))
    return max(1, max(nx, ny)//120)

DS = auto_downsample(nx, ny, args.downsample)

def vorticity(ux, uy):
    uyx, uyy = grad(uy)
    uxx, uxy = grad(ux)
    return uyx - uxy

# ---------------- Physics RHS pieces ----------------
DG = 1.0
DX = params['dx']
DYc = params['dy']
#GammaG = np.exp(-DG * (K2.T) * args.dt)
#GammaX = np.exp(-DX * (K2.T) * args.dt)
#GammaY = np.exp(-DYc * (K2.T) * args.dt)

# reaction terms
b= params['b']; p_par=params['p']; q_par=params['q']; g_par=params['g']; s_par=params['s']; u_cross=params['u_cross']

def reaction_rhs(pG, pX, pY, forcing):
    Xtot = pX + X0
    Ytot = pY + Y0
    nonlinear_s  = s_par * (Xtot**3 - X0**3)
    nonlinear_xy = (Xtot**2 * Ytot - X0**2 * Y0)
    dG = - q_par * pG + g_par * pX
    dX = p_par * pG - (1.0 + b) * pX + u_cross * pY - nonlinear_s + nonlinear_xy + forcing
    dY = b * pX - u_cross * pY + (-nonlinear_xy + nonlinear_s)
    return dG, dX, dY

# velocity RHS (compressible NS)
rho0 = 1.0

def velocity_rhs(ux, uy, pG, pX, pY):
    G = G0 + pG; Xf = X0 + pX; Yf = Y0 + pY
    rho = rho0 + args.alphaG*G + args.alphaX*Xf + args.alphaY*Yf
    rx, ry = grad(np.log(rho + 1e-12))
    ux_x, ux_y = grad(ux)
    uy_x, uy_y = grad(uy)
    convx = ux*ux_x + uy*ux_y
    convy = ux*uy_x + uy*uy_y
    lap_ux = laplacian(ux); lap_uy = laplacian(uy)
    divu = ux_x + uy_y
    divx, divy = grad(divu)
    visc_x = lap_ux + (1.0/3.0)*divx
    visc_y = lap_uy + (1.0/3.0)*divy
    dux = -convx - args.cs2*rx + args.nu*visc_x
    duy = -convy - args.cs2*ry + args.nu*visc_y
    return dux, duy

# semi-Lagrangian advection under periodic BCs

def advect_scalar(phi, ux, uy, dt):
    Xp = (X - dt*ux) % Lx
    Yp = (Y - dt*uy) % Ly
    fx = (Xp / Lx) * nx
    fy = (Yp / Ly) * ny
    i0 = np.floor(fx).astype(int) % nx
    j0 = np.floor(fy).astype(int) % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    sx = fx - np.floor(fx)
    sy = fy - np.floor(fy)
    out = ((1-sx)*(1-sy)*phi[j0, i0] + sx*(1-sy)*phi[j0, i1] + (1-sx)*sy*phi[j1, i0] + sx*sy*phi[j1, i1])
    return out

# ---------------- Rendering ----------------

def render(pG, pX, pY, ux, uy, t, fpath, zlim=1.0, vort_panel=False, quiver=False, quiver_stride=8, live_ax=None):
    step = DS
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    pG_s = pG[::step, ::step]
    pX_s = (pX/10.0)[::step, ::step]
    pY_s = pY[::step, ::step]
    vort = vorticity(ux, uy)
    vort_s = vort[::step, ::step]

    if live_ax is None:
        if vort_panel:
            fig = plt.figure(figsize=(11, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])
            ax3d = fig.add_subplot(gs[0, 0], projection='3d')
            ax2d = fig.add_subplot(gs[0, 1])
        else:
            fig = plt.figure(figsize=(9, 6))
            ax3d = fig.add_subplot(111, projection='3d')
            ax2d = None
    else:
        ax3d = live_ax
        ax3d.clear()
        ax2d = None

    # 3D surface with vorticity colormap on Y
    cmap = cm.coolwarm
    norm = plt.Normalize(vmin=-np.max(np.abs(vort_s)), vmax=np.max(np.abs(vort_s)))
    facecolors = cmap(norm(vort_s))

    ax3d.plot_surface(Xs, Ys, pY_s, facecolors=facecolors, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.85)
    ax3d.plot_surface(Xs, Ys, pG_s, cmap='Blues', alpha=0.5, linewidth=0)
    ax3d.plot_surface(Xs, Ys, pX_s, cmap='Purples', alpha=0.5, linewidth=0)

    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Potential')
    ax3d.set_xlim(0, Lx); ax3d.set_ylim(0, Ly); ax3d.set_zlim(-zlim, zlim)
    ax3d.view_init(elev=35, azim=225)
    ax3d.set_title(f'Model G 2D + Vortices — t={t:.2f}  (ds={step}x)')

    # 2D vorticity panel (optional)
    if ax2d is not None:
        im = ax2d.imshow(vort, origin='lower', extent=[0, Lx, 0, Ly], cmap='coolwarm')
        plt.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04, label='vorticity')
        ax2d.set_xlabel('X'); ax2d.set_ylabel('Y'); ax2d.set_title('Vorticity ωz')
        if quiver:
            qs = quiver_stride
            ax2d.quiver(X[::qs, ::qs], Y[::qs, ::qs], ux[::qs, ::qs], uy[::qs, ::qs], color='k', scale=50)

    if live_ax is None:
        plt.tight_layout(); plt.savefig(fpath, dpi=120); plt.close()
    else:
        ax3d.figure.canvas.draw_idle()

# ---------------- Live viewer ----------------
class LiveViewer:
    def __init__(self, stride=5):
        self.enabled = args.live
        self.stride = max(1, int(stride))
        self.paused = False
        self.quit = False
        self.fig = None
        self.ax = None
        if self.enabled:
            if matplotlib.get_backend().lower() == 'agg':
                try:
                    matplotlib.use('TkAgg')
                except Exception:
                    pass
            plt.ion()
            self.fig = plt.figure(figsize=(9,6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.draw_text()
    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused; self.draw_text()
        elif event.key == 'q':
            self.quit = True
        elif event.key == 's':
            if self.fig is not None:
                self.fig.savefig(os.path.join(out_dir, 'live_snapshot.png'), dpi=140)
                print('[Live] Saved live_snapshot.png')
    def draw_text(self):
        if not self.enabled: return
        self.ax.text2D(0.03, 0.95, '[p] pause/resume  [q] quit  [s] snapshot', transform=self.ax.transAxes)
    def maybe_update(self, frame_idx, pG, pX, pY, ux, uy, t):
        if not self.enabled: return
        if frame_idx % self.stride != 0: return
        while self.paused and not self.quit:
            plt.pause(0.1)
        if self.quit: return
        render(pG, pX, pY, ux, uy, t, None, zlim=args.zlim, vort_panel=args.vort_panel, quiver=args.quiver, quiver_stride=args.quiver_stride, live_ax=self.ax)
        plt.pause(0.001)

live = LiveViewer(stride=args.live_stride)

"""
# ---------------- Time integration (splitting) ----------------
DG = 1.0; DX = params['dx']; DYc = params['dy']
# Recompute spectral decays with shapes aligned to rfft2 output
kx = 2*np.pi*np.fft.fftfreq(nx, d=Lx/nx)
ky = 2*np.pi*np.fft.rfftfreq(ny, d=Ly/ny)
KX2, KY2 = np.meshgrid(kx, ky, indexing='ij')
K2r = KX2**2 + KY2**2
#GammaG = np.exp(-DG * (K2r.T) * args.dt)
GammaX = np.exp(-DX * (K2r.T) * args.dt)
GammaY = np.exp(-DYc * (K2r.T) * args.dt)
"""
frame_times = np.linspace(0.0, args.Tfinal, args.nt_anim)

# ---------------- Checkpoint IO ----------------
def save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done):
    # ✅ Ensure output folder exists before writing temporary file
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    tmp = ckpt_path + '.tmp'
    np.savez_compressed(
        tmp,
        t_curr=t_curr, pG=pG, pX=pX, pY=pY, ux=ux, uy=uy,
        next_frame_idx=next_frame_idx,
        frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
    )
    try:
        os.replace(tmp, ckpt_path)
    except FileNotFoundError:
        print(f"[WARN] Temporary checkpoint {tmp} missing, skipping rename.")


"""
def save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp = ckpt_path + '.tmp'
    np.savez_compressed(
        tmp,
        t_curr=t_curr, pG=pG, pX=pX, pY=pY, ux=ux, uy=uy,
        next_frame_idx=next_frame_idx,
        frames_done=np.array(sorted(list(frames_done)), dtype=np.int32)
    )
    os.replace(tmp, ckpt_path)
"""

def load_ckpt():
    if not os.path.exists(ckpt_path):
        return None
    d = np.load(ckpt_path, allow_pickle=True)
    return {
        't_curr': float(d['t_curr']),
        'pG': d['pG'], 'pX': d['pX'], 'pY': d['pY'], 'ux': d['ux'], 'uy': d['uy'],
        'next_frame_idx': int(d['next_frame_idx']),
        'frames_done': set(int(v) for v in d['frames_done'].tolist())
    }

# core step

def integrate_segment(t0, t1, pG, pX, pY, ux, uy, next_frame_idx, frames_done):
    print("Sanity check → GammaG:", GammaG.shape, "rfft2c(pG):", rfft2c(pG).shape)
    assert GammaG.shape == rfft2c(pG).shape, "GammaG misaligned!"
    
    t = t0
    times_seg = frame_times[(frame_times > t0 + 1e-12) & (frame_times <= t1 + 1e-12)]
    idx_ft = 0
    while t < t1 - 1e-12:
        dt = min(args.dt, t1 - t)
        # 1) Reaction (RK2)
        forcing = chi_xy_t(t)
        dG1, dX1, dY1 = reaction_rhs(pG, pX, pY, forcing)
        pGtmp, pXtmp, pYtmp = pG + dt*dG1, pX + dt*dX1, pY + dt*dY1
        dG2, dX2, dY2 = reaction_rhs(pGtmp, pXtmp, pYtmp, chi_xy_t(t+dt))
        pG += 0.5*dt*(dG1 + dG2); pX += 0.5*dt*(dX1 + dX2); pY += 0.5*dt*(dY1 + dY2)
        # 2) Diffusion (spectral decay)
        print("Shapes -> GammaG:", GammaG.shape, "rfft2c(pG):", rfft2c(pG).shape)
        pG = irfft2c(GammaG * rfft2c(pG))
        pX = irfft2c(GammaX * rfft2c(pX))
        pY = irfft2c(GammaY * rfft2c(pY))
        # 3) Velocity update (RK2)
        dux1, duy1 = velocity_rhs(ux, uy, pG, pX, pY)
        ux_tmp, uy_tmp = ux + dt*dux1, uy + dt*duy1
        dux2, duy2 = velocity_rhs(ux_tmp, uy_tmp, pG, pX, pY)
        ux += 0.5*dt*(dux1 + dux2); uy += 0.5*dt*(duy1 + duy2)
        # mild spectral filter
        Fux = rfft2c(ux); Fuy = rfft2c(uy)
        #filt = np.exp(-0.1 * (K2r.T) * dt)
        filt = np.exp(-0.1 * K2 * dt)
        ux = irfft2c(filt * Fux)
        uy = irfft2c(filt * Fuy)
        # 4) Advect scalars (semi-Lagrangian)
        pG = advect_scalar(pG, ux, uy, dt)
        pX = advect_scalar(pX, ux, uy, dt)
        pY = advect_scalar(pY, ux, uy, dt)
        t += dt
        # render any frames at/within this step
        while idx_ft < len(times_seg) and times_seg[idx_ft] <= t + 1e-12:
            tf = times_seg[idx_ft]
            fidx = np.searchsorted(frame_times, tf)
            if fidx not in frames_done:
                render(pG, pX, pY, ux, uy, tf, os.path.join(frames_dir, f'frame_{fidx:04d}.png'),
                       zlim=args.zlim, vort_panel=args.vort_panel, quiver=args.quiver, quiver_stride=args.quiver_stride)
                frames_done.add(fidx)
                save_ckpt(t, pG, pX, pY, ux, uy, fidx+1, frames_done)
                live.maybe_update(fidx, pG, pX, pY, ux, uy, tf)
            idx_ft += 1
    return t, pG, pX, pY, ux, uy, next_frame_idx, frames_done

# ---------------- Main ----------------
class Live:
    pass

live = LiveViewer(stride=args.live_stride)
#live = Live()
live.viewer = None

class LiveViewer:
    def __init__(self, stride=5):
        self.enabled = args.live
        self.stride = max(1, int(stride))
        self.paused = False
        self.quit = False
        self.fig = None
        self.ax = None
        if self.enabled:
            if matplotlib.get_backend().lower() == 'agg':
                try:
                    matplotlib.use('TkAgg')
                except Exception:
                    pass
            plt.ion()
            self.fig = plt.figure(figsize=(9,6))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.draw_text()
    def on_key(self, event):
        if event.key == 'p':
            self.paused = not self.paused; self.draw_text()
        elif event.key == 'q':
            self.quit = True
        elif event.key == 's':
            if self.fig is not None:
                self.fig.savefig(os.path.join(out_dir, 'live_snapshot.png'), dpi=140)
                print('[Live] Saved live_snapshot.png')
    def draw_text(self):
        if not self.enabled: return
        self.ax.text2D(0.03, 0.95, '[p] pause/resume  [q] quit  [s] snapshot', transform=self.ax.transAxes)
    def maybe_update(self, frame_idx, pG, pX, pY, ux, uy, t):
        if not self.enabled: return
        if frame_idx % self.stride != 0: return
        while self.paused and not self.quit:
            plt.pause(0.1)
        if self.quit: return
        render(pG, pX, pY, ux, uy, t, None, zlim=args.zlim, vort_panel=args.vort_panel, quiver=args.quiver, quiver_stride=args.quiver_stride, live_ax=self.ax)
        plt.pause(0.001)

live.viewer = LiveViewer(stride=args.live_stride)

def main():
    # resume or start
    if os.path.exists(ckpt_path):
        d = np.load(ckpt_path, allow_pickle=True)
        t_curr = float(d['t_curr'])
        pG = d['pG']; pX = d['pX']; pY = d['pY']; ux = d['ux']; uy = d['uy']
        next_frame_idx = int(d['next_frame_idx'])
        frames_done = set(int(v) for v in d['frames_done'].tolist())
        print(f"[Resume] t={t_curr:.3f}, frames_done={len(frames_done)}")
    else:
        t_curr = 0.0
        pG = np.zeros((ny, nx)); pX = np.zeros((ny, nx)); pY = np.zeros((ny, nx))
        ux = np.zeros((ny, nx)); uy = np.zeros((ny, nx))
        if args.rotseed:
            cx = args.swirl_cx if args.swirl_cx is not None else Lx/2
            cy = args.swirl_cy if args.swirl_cy is not None else Ly/2
            dxg = X - cx; dyg = Y - cy
            r2 = dxg*dxg + dyg*dyg
            vtheta = args.swirl_amp * np.exp(-r2 / (2*args.swirl_sigma**2))
            R = np.sqrt(r2) + 1e-12
            ux = - vtheta * (dyg / R)
            uy =   vtheta * (dxg / R)
        next_frame_idx = 0
        frames_done = set()
        print("[Start] Fresh run")

    # pre-render t=0
    if next_frame_idx < args.nt_anim and frame_times[next_frame_idx] <= t_curr + 1e-12:
        render(pG, pX, pY, ux, uy, t_curr, os.path.join(frames_dir, f'frame_{next_frame_idx:04d}.png'),
               zlim=args.zlim, vort_panel=args.vort_panel, quiver=args.quiver, quiver_stride=args.quiver_stride)
        frames_done.add(next_frame_idx); next_frame_idx += 1
        save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done)

    # integrate in segments
    t_start = time.time()
    while t_curr < args.Tfinal - 1e-12 and len(frames_done) < min(args.nt_anim, args.max_frames) and not live.viewer.quit:
        t_seg_end = min(args.Tfinal, t_curr + args.segment_dt)
        print(f"[Integrate] {t_curr:.3f} -> {t_seg_end:.3f}  (dt={args.dt})")
        t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done = integrate_segment(
            t_curr, t_seg_end, pG, pX, pY, ux, uy, next_frame_idx, frames_done)
        render(pG, pX, pY, ux, uy, t_curr, final_png,
               zlim=args.zlim, vort_panel=args.vort_panel, quiver=args.quiver, quiver_stride=args.quiver_stride)
        save_ckpt(t_curr, pG, pX, pY, ux, uy, next_frame_idx, frames_done)
        print(f"  -> t={t_curr:.3f}/{args.Tfinal}, frames={len(frames_done)}/{args.nt_anim}, wall={time.time()-t_start:.1f}s")

    # MP4 assembly
    print('[Video] Writing MP4:', mp4_path)
    with imageio.get_writer(mp4_path, fps=max(8, int(args.nt_anim / max(1, args.Tfinal/2)))) as w:
        for i in range(min(args.nt_anim, args.max_frames)):
            f = os.path.join(frames_dir, f'frame_{i:04d}.png')
            if os.path.exists(f):
                w.append_data(imageio.imread(f))
    print('[Done] MP4 saved.')

if __name__ == '__main__':
    main()

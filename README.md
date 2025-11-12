ChatGPT:

I added a new ready-to-run script, model_g_2d_xy_vortical_plus__2b.py, with everything you asked for:

New features

Vorticity panel (optional) next to the 3D surface.

Quiver overlay (optional) on the vorticity panel to show the velocity field arrows.

Fluid parameters exposed on CLI: --alphaG/--alphaX/--alphaY, --cs2, --nu, plus all Model-G kinetics.

Rotating seed preset (--rotseed) to kickstart coherent vortices. You can control center/strength/size via --swirl_amp, --swirl_sigma, --swirl_cx, --swirl_cy.

Keeps safe + resumable design (checkpoints + MP4).

Uses the FFT pseudo-spectral solver (periodic BCs), consistent with the vortical approach.

How to run

Batch (with vortices, vorticity panel + quiver):

python3 model_g_2d_xy_vortical_plus__2b.py \
  --nx 192 --ny 192 --Lx 60 --Ly 60 \
  --Tfinal 8 --dt 0.005 --segment_dt 0.5 --zlim 1.0 \
  --alphaG 0.02 --alphaX 0.02 --alphaY 0.02 --cs2 1.0 --nu 0.25 \
  --rotseed --swirl_amp 1.0 --swirl_sigma 6.0 \
  --vort_panel --quiver --quiver_stride 8


Live viewer (needs interactive backend; install one if needed: sudo apt install python3-tk or set MPLBACKEND=TkAgg):

MPLBACKEND=TkAgg python3 model_g_2d_xy_vortical_plus__2b.py --live --rotseed --vort_panel --quiver

Notes & tips

Z-axis range is fixed via --zlim (default ±1.0) so your surfaces aren’t swamped.

The vorticity used to color the Y surface is ωₙ = ∂ₓu_y − ∂ᵧu_x.

The quiver overlay is downsampled with --quiver_stride (increase for fewer arrows).

For stronger vortex rings, bump --swirl_amp and/or reduce --swirl_sigma; move the swirl with --swirl_cx/--swirl_cy.

Outputs: out_model_g_2d_xy_vortical_plus/frames/, .../final_snapshot.png, .../checkpoint_vortical_plus.npz, and .../model_g_2d_xy_vortical_plus.mp4.

If you hit anything odd, tell me the error text and the exact command you ran, and I’ll patch it right away. Want me to port the same
vorticity panel + quiver options back into your non-fluid 2D solver too, so you can compare “with vs without flow” side-by-side?

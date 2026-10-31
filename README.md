# Dual-arm block stacking

`branch_w` implements visual ACT-CVAE and Diffusion Policy offline training,
ACT residual PPO and diffusion denoising-chain PPO online training, remote Isaac
simulation, and checkpoint-backed RoboDojo evaluation adapters.

The runnable project remains in `startcode/`; its name is preserved for existing
course commands. See [训练与仿真完整说明](docs/TRAINING_AND_SIMULATION.md) for every
file's role, core code, server setup, exact commands, tests, and limitations.

Small CPU tests use synthetic data and do not demonstrate stacking performance.
Real Isaac rendering, GPU memory use, and task success must be checked on the server.

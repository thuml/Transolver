"""

Usage of Transolver_Structured_Mesh_2D.py

"""

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

#!-----------------------------------------------------------------------
class Model(nn.Module):

#!--------------------------------------------
   def get_grid(self, batchsize=1):
   -> Capture the relative geometry
   -> Have more features
   -> reference grid is self.ref * self.ref
   -> mesh grid is self.H * self.W

        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2
        -> Concatenate them along the last dimension
        -> Example:
            -> gridx.shape = [B, H, W, 1]
            -> gridy.shape = [B, H, W, 1]
            -> grid.shape  = [B, H, W, 2]
        -> We can get point-wise pair by grid[0, x_i, y_i]

#!---------------------
        diff = grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]
        -> grid[:, :, :, None, None, :]
           -> Pytorch automatically broadcasts the tensor "grid" shape into [B, H, W, ref, ref, 2]
           -> Now the channel is still two
        -> grid_ref[:, None, None, :, :, :]
           -> Same operation
           -> grid_ref.shape = [B, H, W, ref, ref, 2]
        -> Now "grid" and "grid_ref" have the same shape, and are okey to have some math opreation
        -> diff.shape = [B, H, W, ref, ref, 2]
           -> "diff" stores dx and dy between mesh points and reference points

#!---------------------
        torch_sum = torch.sum(diff**2, dim=-1)
        -> torch_sum.shape = [B, H, W, ref, ref]
           -> It stores dx^2 + dy^2

#!---------------------
        pos = sqrt.reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        -> pos.shape = [B, H, W, ref*ref]
           -> H: mesh-grid row
           -> W: mesh-grid column
           -> ref*ref: flattened reference-grid index
        -> How to access the distance beween mehs_point(aa, bb) and ref_point(ii, jj)
           -> ref_idx = ii * self.ref + jj
           -> {B, aa, bb, ref_idx}

#!--------------------------------------------
    def forward(self, x, fx, T=None):

#!---------------------
    if self.unified_pos:
        x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
    -> This snippet generates a unified mesh grid to overwrite the previous loaded in "exp_darcy.py"

#!---------------------
    fx = torch.cat((x, fx), -1)
    -> Concatenates along the last dimension
       -> x.shape = [B, N, 64], fx.shape = [B, N, 1]
       -> fx.shape = [B, N, 65]
    -> Example:
       -> fx[0, 0, 0]
          -> At batch 0, the distance between mesh grid "0" and reference point "0"
       -> fx[0, 0, 64]
          -> At batch 0, the pressure value at mesh grid point "0"














#!-----------------------------------------------------------------------
class MLP(nn.Module):
    -> There exists pre and post for act()
    -> MLP is not a linear neural network

#!--------------------------------------------
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
    -> Example with n_layers
        -> Keyword argu: MLP(n_layers=0)
        -> Positional argu: MLP(0)
        -> If we use Keyword argu, the passing order can be mixed
        -> Positional argu is permitted

#!---------------------
    -> self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
       -> Sequential is a intermediate process between Input and Output
       -> pipeline for neutral network
          -> Input -> Layer1 -> Activation -> Layer2 -> ... -> Output

#!---------------------
    -> self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
       -> Build a list of "n_layers"
       -> Store the multi-stacked Sequential blocks



""" variable meaning """
# Input=fun_dim + self.ref^2          (pos + physical input)
# fun_dim:                            number of physical features per mesh points(e.g. velocity, pressure)
# self.ref:                           reference grid points
# num_head:

""" Transolver_block() passing arguments meaning """
# num_heads


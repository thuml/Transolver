"""

Usage of Transolver_Structured_Mesh_2D.py

"""

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

#!-----------------------------------------------------------------------
class Model(nn.Module):

#!-----------------------------------------------------------------------
class MLP(nn.Module):

#!---------------------
   def get_grid(self, batchsize=1):

        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2
        -> Concatenate them along the last dimension
        -> Example:
            -> gridx.shape = [B, H, W, 1]
            -> gridy.shape = [B, H, W, 1]
            -> grid.shape  = [B, H, W, 2]



""" Variable meaning """
# fun_dim:              number of physical features per mesh points(e.g. velocity, pressure)
# self.ref:             refinement
# Input=fun_dim + self.ref^2 (pos + physical input)

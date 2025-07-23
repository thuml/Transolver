"""

Usage of Physics_Attention.py

"""

#!-----------------------------------------------------------------------
class Physics_Attention_Structured_Mesh_2D(nn.Module):

""" Physics_Attention_Structured_Mesh_2D() passing arguments meaning """
#! dim:                                Input feature dimension per mesh point(e.g. coordinate and perssure in Darcy flow)
#! heads:                              Number of attention heads.
  -> The model splits the intermediate features into multiple independent subspaces.
  -> Each head capture different types.
#! dim_head:                           Size of subspace each slice token projected.
#! inner_dim:                          Total featrue dimension after projecting input data for attention
#! slice_num:                          Number of learnable slice tokens.
  -> It defines how many groups the mesh will be assigned to
  -> e.g. slice_num=64
     -> Each mesh point is assigned to these 64 slices via softmax weights


#!--------------------------------------------
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
    -> built-in function

#!---------------------
    self.softmax = nn.Softmax(dim=-1)
    -> Softmax() converts a vector of real numbers into a "probability distribution"
       -> It is defined as:
          ->"$softmax(x_{i})=\frac{e^{x_{i}}}{\sum_{j}e^{x_{j}}}$"
       -> In Physics-Attention
          -> We wanna softly assign each mesh point to multiple physics-aware slices
          -> Apply "softmax()" to get each slice wights
          -> So that each point is distributed across slices
             -> e.g. 0.7 to slice 1, 0.2 to slice 2, etc.
          -> That is how model learns data-dependent, differentiable groupings
    -> dim=-1
       -> find weight along tyhe last dimension of the tensor

#!---------------------
    self.dropout = nn.Dropout(dropout)
    -> Dropout() is a regularization technique used during "training" to prevent overfitting
       -> It randomly sets some elements of the input tensor to "zero", with a given probability
       -> And the rest are scaled by "1 / (1-dropout)" to preserve the expected value

#!---------------------
    self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
    -> It defines a learnable temperature used in the softmax() func that generates the "slice weights"
    -> In general
       -> softmax(x/T)
       -> x is the raw score
       -> T > 1: output is softer, more uniform
       -> T < 1: output is sharper, more confident
    -> nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
       -> Start T with 0.5
       -> nn.Parameter()
          -> Make temperature learable
       -> self.temperature.shape = [1, heads, 1, 1]
          -> It is compatibale with input_data

#!---------------------
    self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
    -> Each output channe is a weighted combination of local patches of input_data
    -> nn.Conv2d(in_channels=dim, out_channels=inner_dim, kernel_size=kernel, stride=1, padding=kernel // 2)
       -> kernel_size
          -> This sets the size of "sliding widow" (also called filter) that the convolution uses to exrtact local feature
          -> kernel_size=3
             -> the filter looks at a 3*3 square
          -> To a physical problem, pressure , flux, or something
             -> The value at a point is affected by its neighbors
             -> Example: Pressure at a point is influnced by the gradient
                -> We need the nearby points message
             -> So the kernel_size > 1
       -> stride
          -> It defines how far the filter moves each time it slides over the mesh
          -> stride = 1
             -> Full resolution.
             -> Recommended for physics
       -> padding
          -> This adds zeros or another value around the border of the input_data
             -> Make sure the output mesh size stays the same with input mesh size
             -> i.e. output.shape = [B,c_out , H, W]
                -> H and W is the same as input_data
          -> padding_size = kernel_size / 2
             -> It depends the kernel_size, not personal








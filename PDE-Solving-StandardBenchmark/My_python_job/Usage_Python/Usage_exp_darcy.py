"""
    Usage of exp_darcy.py

    Darcy Flow

    Coeff perimeable coefficient
    Sol   pressure value

"""
def main():
    train_data = scio.loadmat(train_path)
    -> Load the .mat file

#!------------------------ How to checkout a black .mat life
    import scipy.io as scio
    train_data = scio.loadmat(train_path)
    logging.info(f"{M} trian_data: {train_data.keys()} {RESET}")
    logging.info(f"{M} coeff value at point [i, 1, 1]: {tmp_3[1, 1, 1]} {RESET}")
    -> The last two slot is the postion index
    -> It is "2D" geometry !!!


#!------------------------
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    -> x_train.shape = (1024, 421, 421)
       -> The first slot means the number of samples
    -> [:ntrain, ::r, ::r]
       -> :ntrain
          -> Number of train samples
       -> ::r
          -> take every r-th element
          -> be equivalent to  0:end:r
    -> [:, :s, :s]
       -> :
          -> take all data
       -> :s
          -> take the first "s" data

#!------------------------
    x_train = x_train.reshape(ntrain, -1)
    -> .reshape(ntrain, -1)
       -> -1: Flattening the remaining dimensions
       -> (1000, s, s)
       -> (1000, s*s)

#!------------------------
    x_normalizer = UnitTransformer(x_train)
    -> x_normalizer is an object of class UnitTransformer

#!------------------------
    x_train = x_normalizer.encode(x_train)
    -> .encode()
       -> normalize the data

#!------------------------
    x = np.linspace(0, 1, s)
    -> Create a "1D" array of evenly spaced
    -> np.linspace(start, stop, num)
    -> Example:
       -> s = 5
       -> (0 ,1/4, 2/4, 3/4, 4/4)

#!------------------------
    x, y = np.meshgrid(x, y)
    -> Create a "2D" mesh grid
    -> x and y both stores "(s,s)"

#!------------------------
    pos = np.c_[x.ravel(), y.ravel()]
    -> Flatten x and y grid coor and stack them into an array of shape
       Each row is an "(x,y)" points
    -> np.c_[]
       -> Concatenate array "column-wise"
       -> Output: (s*s, 2)
       -> Example:
           x.ravel() = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
           y.ravel() = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

          pos =
          [[0.0, 0.0],
           [0.5, 0.0],
           [1.0, 0.0],
           [0.0, 1.0],
           [0.5, 1.0],
           [1.0, 1.0]]


    -> x.ravel()
       -> Flatten a multi-D array into "1D" array
       -> the remaining rows all are put afer the first row line by line
       -> Example:
          -> x =
                [[0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0]]

             x.ravel() = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]

#!------------------------
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    -> make Numpy array a PyTorch Tensor
    -> .unsqueeze(0)
       -> add new dimension for the first slot
       -> pos.shape = (batch_size, num_points, feature)

#!------------------------
    pos_train = pos.repeat(ntrain, 1, 1)
    -> .repeat(a, b, c, ...)
       -> Repeat a, b, c,... specify how many times to repeat along each dimension
    -> pos.shape = (ntrain, num_points, feature)

#!------------------------
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    -> Wrap these threee tensore as a dataset
    -> shffle = True
       -> Shuffle the dataset at the beginning of every epoch
       -> Ensure the model does not see samples in the same order every time

#!------------------------
    for x, fx, y in tqdm(train_loader, desc="[Training]"):
    -> x  is pos_train, stands for mesh points
    -> fx is x_train, the physical value, e.g. Pressure
    -> y  is y_train

#!------------------------
    out = out[..., 1:-1, 1:-1].contiguous()
    -> Prepare for central difference method
       -> out.shape = [b c h w]
       -> In 'h-dimension' remove the first and last row
       -> In 'w-dimension' remove the first and lats row

#!------------------------
    out = F.pad(out, (1, 1, 1, 1), "constant", 0)
    -> add a specified value to the "tensor" out
    -> pad = [left, right, top, bottom]
       -> (1, 1, 1, 1)
       -> Pad 1 column on the left
       -> ...
    -> Example:
       -> Before pad, out.shape = [2, 3, 4, 4]
       -> After pad,  out.shape = [2, 3, 6, 6]




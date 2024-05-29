from model import Transolver_1D, Transolver_2D, Transolver_3D


def get_model(args):
    model_dict = {
        'Transolver_1D': Transolver_1D, # for PDEs in 1D space or in unstructured meshes
        'Transolver_2D': Transolver_2D,
        'Transolver_3D': Transolver_3D,
    }
    return model_dict[args.model]


import pickle
import torch

"""
Basic utility functions for saving.
"""

def save_object(object, output_location: str, object_name: str, epoch: int = -1, \
                iteration: int = -1) -> None:
    if epoch >= 0 and iteration >= 0:
        _append_str = output_location + object_name + 'ep' + str(epoch) + 'iter' + str(iteration)
    else:
        _append_str = output_location + object_name
                        
    print(f'Saving to {_append_str}.pkl')

    file = open(_append_str + '.pkl', 'wb')
    pickle.dump(object, file)
    file.close()
    print('Done!')

def save_model(output_location: str, model: torch.nn.Module, iteration: int) -> None:
    torch.save({'iteration': iteration,
               'model_state_dict': model.state_dict(),
               }, output_location + str(iteration) + "it.pt")
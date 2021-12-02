
import pickle

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

    file = open(output_location, 'wb')
    pickle.dump(object, file)
    file.close()
    print('Done!')

def save_model(output_location: str, model) -> None:
    pass
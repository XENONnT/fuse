import numpy as np
import epix


def full_array_to_numpy(array, dtype):
    
    len_output = len(epix.awkward_to_flat_numpy(array["x"]))

    numpy_data = np.zeros(len_output, dtype=dtype)

    for field in array.fields:
        numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        
    return numpy_data
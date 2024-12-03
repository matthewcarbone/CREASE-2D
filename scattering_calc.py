import pandas as pd
import cupy as cp
import numpy as np
import math
from time import time

# Load data
col_types = {'item': 'float32', 'atoms': 'float32', 'id': 'float32', 'type': 'float32', 'x': 'float32', 'y': 'float32', 'z' : 'float32'}
data = pd.read_csv("samplestructure.dump", delim_whitespace=True, skiprows=9, header=None, usecols=[0, 1, 2, 3, 4, 5], dtype=col_types)
boxdim = pd.read_csv("samplestructure.dump", delim_whitespace=True, dtype=col_types, header=None, skiprows=5, nrows=3)

# Convert DataFrame to CuPy array for GPU computation
data_cp = cp.array(data.values)
boxdim_cp = cp.array(boxdim.values)

# Constants
ntheta, qmin_exp, qmax_exp, nq = 91, -2, 3, 501

# Function to process each chunk
def process_chunk(chunk_id, subXYZ, dir1value, dir2value, qmagvalue, boxformfactorxy, boxformfactoryz, boxformfactorxz):
    start_time = time()
    currentchunksize = subXYZ.shape[0]
    qposxy = (qmagvalue @ cp.ones((1, currentchunksize))) * ((dir1value @ subXYZ[:, 0].reshape(1, -1)) + (dir2value @ subXYZ[:, 1].reshape(1, -1)))
    qposyz = (qmagvalue @ cp.ones((1, currentchunksize))) * ((dir1value @ subXYZ[:, 1].reshape(1, -1)) + (dir2value @ subXYZ[:, 2].reshape(1, -1)))
    qposxz = (qmagvalue @ cp.ones((1, currentchunksize))) * ((dir1value @ subXYZ[:, 0].reshape(1, -1)) + (dir2value @ subXYZ[:, 2].reshape(1, -1)))

    resultxy = cp.sum(cp.exp(-1j * qposxy) - boxformfactorxy * cp.ones((1, currentchunksize), dtype=cp.float32), axis=1)
    resultyz = cp.sum(cp.exp(-1j * qposyz) - boxformfactoryz * cp.ones((1, currentchunksize), dtype=cp.float32), axis=1)
    resultxz = cp.sum(cp.exp(-1j * qposxz) - boxformfactorxz * cp.ones((1, currentchunksize), dtype=cp.float32), axis=1)

    elapsed_time = time() - start_time
    print(f"Chunk ID: {chunk_id}, Processed in {elapsed_time:.2f} seconds.")
    return resultxy, resultyz, resultxz

# Main calculation function with batch processing
def gpu_calculations_parallel(data_cp, boxdim_cp):
    # Pre-calculation setup (boxlen, boxrad, qgrid, thetagrid, qmag, dir1grid, dir2grid, etc.)
    boxlen = cp.mean(boxdim_cp[:, 1] - boxdim_cp[:, 0])
    boxrad = boxlen / 2
    qgrid = cp.logspace(qmin_exp, qmax_exp, nq, dtype=cp.float32).reshape(-1,1) @ cp.ones((1, ntheta), dtype=cp.float32)
    thetagrid = cp.ones((nq, 1), dtype=cp.float32) @ cp.linspace(0, cp.pi / 2, ntheta, dtype=cp.float32).reshape(1,-1)
    qmag = cp.repeat(cp.expand_dims(qgrid, axis=2), 2, axis=2)
    qmag[:, :, 1] = qmag[:, :, 0]
    qmag = np.transpose(qmag, (2, 0, 1))
    dir1grid = cp.repeat(cp.expand_dims(cp.cos(thetagrid), axis=2), 2, axis=2)
    dir2grid = cp.repeat(cp.expand_dims(cp.sin(thetagrid), axis=2), 2, axis=2)
    dir1grid[:, :, 1] = -dir1grid[:, :, 0]
    dir2grid[:, :, 1] = dir2grid[:, :, 0]
    a = cp.transpose(dir1grid, (2,0,1))
    b = cp.transpose(dir2grid, (2,0,1))
    a1 = a[0,:,:]
    a2 = a[1,:, :]
    a1 = cp.transpose(a1)
    a2 = cp.transpose(a2)
    a1_ele = a1.reshape(-1,1)
    a2_ele = a2.reshape(-1,1)
    dir1value = cp.concatenate((a1_ele, a2_ele), axis = 0)
    b1 = b[0,:,:]
    b2 = b[1,:, :]
    b1 = cp.transpose(b1)
    b2 = cp.transpose(b2)
    b1_ele = b1.reshape(-1,1)
    b2_ele = b2.reshape(-1,1)
    dir2value = cp.concatenate((b1_ele, b2_ele), axis = 0)
    qmag1 = qmag[0, :,:]
    qmag2 = qmag[1,:,:]
    qmag1 = cp.transpose(qmag1)
    qmag2 = cp.transpose(qmag2)
    ele_qmag1 = qmag1.reshape(-1,1)
    ele_qmag2 = qmag2.reshape(-1,1)
    qmagvalue = cp.concatenate((ele_qmag1, ele_qmag2), axis = 0)
    boxformfactorxy = 3 * (cp.sin(qmagvalue * boxrad) - (qmagvalue * boxrad) * cp.cos(qmagvalue * boxrad)) / (qmagvalue * boxrad) ** 3
    boxformfactoryz = 3 * (cp.sin(qmagvalue * boxrad) - (qmagvalue * boxrad) * cp.cos(qmagvalue * boxrad)) / (qmagvalue * boxrad) ** 3
    boxformfactorxz = 3 * (cp.sin(qmagvalue * boxrad) - (qmagvalue * boxrad) * cp.cos(qmagvalue * boxrad)) / (qmagvalue * boxrad) ** 3

    # Prepare XYZ data
    XYZ = cp.column_stack((data_cp[:, 3], data_cp[:, 4], data_cp[:, 5]))
    XYZ = cp.mod(XYZ + boxrad, boxlen) - boxrad
    datamap = cp.sum(XYZ ** 2, axis=1) < boxrad ** 2
    XYZ = XYZ[datamap, :]
    Numbeads = XYZ.shape[0]

    # Set batch size and calculate number of batches
    batch_size = 1000  # Adjust based on your GPU memory
    num_batches = math.ceil(Numbeads / batch_size)


    chunksize = 256
    nchunks = math.ceil(Numbeads / chunksize)
    nchunks = 2 ** math.ceil(math.log2(nchunks))
    chunksize = math.ceil(Numbeads / nchunks)
    numpaddedvals = nchunks * chunksize - Numbeads
    lastvalidchunk = nchunks - cp.floor((nchunks * chunksize - Numbeads) / chunksize)
    XYZ_padded = cp.pad(XYZ, ((0, numpaddedvals), (0, 0)), mode='constant')
    chunkXYZ = (XYZ_padded.reshape(nchunks,chunksize , 3))

    # Initialize results
    Ampxy = Ampyz = Ampxz = 0

    # Process each batch
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, Numbeads)
        subXYZ = XYZ[start_idx:end_idx]

        resultxy, resultyz, resultxz = process_chunk(batch, subXYZ, dir1value, dir2value, qmagvalue, boxformfactorxy, boxformfactoryz, boxformfactorxz)
        Ampxy += resultxy
        Ampyz += resultyz
        Ampxz += resultxz

    return Ampxy, Ampyz, Ampxz

# Run the GPU calculations with batch processing
Ampxy, Ampyz, Ampxz = gpu_calculations_parallel(data_cp, boxdim_cp)

# Post-calculation processing (reshape and concatenate data)
resultqxqy = cp.log10(cp.abs(Ampxy * cp.conj(Ampxy)))
resultqyqz = cp.log10(cp.abs(Ampyz * cp.conj(Ampyz)))
resultqxqz = cp.log10(cp.abs(Ampxz * cp.conj(Ampxz)))
resultqxqy = resultqxqy.reshape(2, 91, 501)
resultqxqy1 = cp.transpose(resultqxqy[0])
resultqxqy2 = cp.transpose(resultqxqy[1])
resultqyqz = resultqyqz.reshape(2, 91, 501)
resultqyqz1 = cp.transpose(resultqyqz[0])
resultqyqz2 = cp.transpose(resultqyqz[1])
resultqxqz = resultqxqz.reshape(2, 91, 501)
resultqxqz1 = cp.transpose(resultqxqz[0])
resultqxqz2 = cp.transpose(resultqxqz[1])

# Concatenate and flip as in MATLAB
dataxy = cp.concatenate([resultqxqy1[:, :-1], cp.fliplr(resultqxqy2[:, :])], axis=1)
datayz = cp.concatenate([resultqyqz1[:, :-1], cp.fliplr(resultqyqz2[:, :])], axis=1)
dataxz = cp.concatenate([resultqxqz1[:, :-1], cp.fliplr(resultqxqz2[:, :])], axis=1)


# Save results
pd.DataFrame(cp.asnumpy(dataxy)).to_csv('samplestructure_dataxy.txt', header=False, index=False)
pd.DataFrame(cp.asnumpy(datayz)).to_csv('samplestructure_datayz.txt', header=False, index=False)
pd.DataFrame(cp.asnumpy(dataxz)).to_csv('samplestructure_dataxz.txt', header=False, index=False)
print("Data saved successfully.")
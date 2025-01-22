import h5py
import numpy as np

# Define the GPS start time
gps_start_time = 1264314069  # Example GPS time

# File path
file_path = "GW200129_065458_1264314069.hdf5"

# Open the HDF5 file in read/write mode
with h5py.File(file_path, "a") as f:
    # Check if "GPS_start_time" dataset exists, and if not, create it
    if "GPS_start_time" not in f:
        # Create a dataset for GPS_start_time similar to the "merger" datasets
        f.create_dataset("GPS_start_time", data=np.array([gps_start_time], dtype="i8"))

# Verify the addition by reading the dataset
with h5py.File(file_path, "r") as f:
    print("GPS_start_time dataset:", f["GPS_start_time"][0])


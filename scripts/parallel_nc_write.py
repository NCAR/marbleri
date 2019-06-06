import xarray as xr
import numpy as np
from multiprocessing import Pool, Manager
from os.path import exists
from dask.distributed import Lock, LocalCluster, Client


def main():
    #p = Pool(2)
    #manager = Manager()
    #lock = manager.Lock()
    cluster = LocalCluster(4)
    client = Client(cluster)
    futures = []
    for i in range(30):
        futures.append(client.submit(p_write, i))
    client.gather(futures)
    client.close()
    return


def p_write(v):
    print("Start", v)
    y = 10
    x = 15
    lock = Lock("x")
    data = np.ones((y, x), dtype=np.float32) * v
    da = xr.DataArray(data, dims=("y", "x"),
                      coords={"y": np.arange(y), "x": np.arange(x),
                                    "lon": xr.DataArray(np.arange(x) + v, dims=("x",)),
                              "lat": xr.DataArray(np.arange(y) + v, dims=("y",))}, name=f"test_{v:02d}")
    lock.acquire()
    print("output", v)
    if not exists("test.nc"):
        mode = "w"
    else:
        mode = "a"
    da.to_netcdf("test.nc", mode=mode)
    lock.release()

if __name__ == "__main__":
    main()
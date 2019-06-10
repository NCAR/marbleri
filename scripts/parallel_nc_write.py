import xarray as xr
import numpy as np
from multiprocessing import Pool, Manager
from os.path import exists
from dask.distributed import Lock, LocalCluster, Client, Queue


def main():
    #p = Pool(2)
    #manager = Manager()
    #lock = manager.Lock()
    num_procs = 4
    cluster = LocalCluster(num_procs)
    client = Client(cluster)
    futures = []
    queue = Queue("x", client=client)
    done_sum = 0
    for i in range(0, 500, 50):
        futures.append(client.submit(p_write, i))
    while True:
        if queue.qsize() > 0:
            queue_out = queue.get()
            print(queue_out)
            if queue_out == 1:
                done_sum += 1
                if done_sum == num_procs:
                    break
            else:
                print(queue_out[0], queue_out[1])

    client.close()
    return


def p_write(v):
    print("Start", v)
    y = 10
    x = 15
    queue = Queue("x")
    for i in range(v, v + 50):
        data = np.ones((y, x), dtype=np.float32) * v
        da = xr.DataArray(data, dims=("y", "x"),
                      coords={"y": np.arange(y), "x": np.arange(x),
                                    "lon": xr.DataArray(np.arange(x) + v, dims=("x",)),
                              "lat": xr.DataArray(np.arange(y) + v, dims=("y",))}, name=f"test_{v:02d}")
        queue.put(data)
    queue.put(1)
    #lock.acquire()
    #print("output", v)
    #if not exists("test.nc"):
    #    mode = "w"
    #else:
    #    mode = "a"
    #da.to_netcdf("test.nc", mode=mode)
    #lock.release()


if __name__ == "__main__":
    main()
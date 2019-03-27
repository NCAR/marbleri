from marbleri.nwp import BestTrackNetCDF


def main():
    # load best track data
    bt_nc = BestTrackNetCDF()
    # convert best track data to data frame and filter out NaNs in HWRF and best track winds

    # calculate derived variables in data frame

    # in parallel extract variables from each model run, subset center from rest of grid and save to other
    # netCDF files
    return


if __name__ == "__main__":
    main()

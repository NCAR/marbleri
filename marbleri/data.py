import numpy as np
from os.path import join, exists
from os import makedirs

def output_preds_adeck(pred_df, best_track_df, model_name, model_tech_code, out_path, delimiter=","):
    """
    Output ML predictions to ATCF best track format. Format definition at
    https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt.
    More information about the ATCF tropical cyclone database at
    https://www.nrlmry.navy.mil/atcf_web/docs/database/new/database.html.

    Args:
        pred_df:
        best_track_df:
        out_path:

    Returns:

    """
    adeck_columns = ["BASIN",
                    "CY",
                    "YYYYMMDDHH",
                    "TECHNUM/MIN",
                    "TECH",
                    "TAU",
                    "LatN/S",
                    "LonE/W",
                    "VMAX",
                    "MSLP",
                    "TY",
                    "RAD",
                    "WINDCODE",
                    "RAD1",
                    "RAD2",
                    "RAD3",
                    "RAD4",
                    "POUTER",
                    "ROUTER",
                    "RMW",
                    "GUSTS",
                    "EYE",
                    "SUBREGION",
                    "MAXSEAS",
                    "INITIALS",
                    "DIR",
                    "SPEED",
                    "STORMNAME",
                    "DEPTH",
                    "SEAS",
                    "SEASCODE",
                    "SEAS1",
                    "SEAS2",
                    "SEAS3",
                    "SEAS4"]
    adeck_chars = [2,
                  2,
                  10,
                  2,
                  4,
                  3,
                  4,
                  5,
                  3,
                  4,
                  2,
                  3,
                  3,
                  4,
                  4,
                  4,
                  4,
                  4,
                  4,
                  3,
                  3,
                  3,
                  3,
                  3,
                  3,
                  3,
                  3,
                  10,
                  1,
                  2,
                  3,
                  4,
                  4,
                  4,
                  4]
    # convert best track basins to ADECK basins
    basin_id = {"l": "AL",
                "e": "EP",
                "c": "CP",
                "w": "WP",
                "p": "SH",
                "s": "SH",
                "b": "IO",
                "a": "IO"}
    if not exists(out_path):
        makedirs(out_path)
    stnum_str = best_track_df["STNUM"].astype(str).str.zfill(2)
    date_str = best_track_df["DATE"].astype(str).str[:4]
    basin_num_year = best_track_df["BASIN"] + stnum_str + date_str
    storms = np.unique(basin_num_year)
    for storm in storms:
        adeck_storm_id = storm.replace(storm[0], basin_id[storm[0]])
        print(adeck_storm_id)
        storm_idxs = basin_num_year == storm
        pred_storm = pred_df.loc[storm_idxs]
        bt_storm = best_track_df.loc[storm_idxs]
        vmax_curr = int(bt_storm.iloc[0]["VMAX"] - bt_storm.iloc[0]["VMAX_dt_24"])
        with open(join(out_path, adeck_storm_id + ".dat"), "w") as adeck_file:
            for i in range(pred_storm.shape[0]):
                out_list = []
                #BASIN
                out_list.append(basin_id[bt_storm.iloc[i]["BASIN"]])
                # CY
                out_list.append(str(bt_storm.iloc[i]["STNUM"]).zfill(2))
                # YYYYMMDDHH
                out_list.append(str(bt_storm.iloc[i]["DATE"]))
                # "TECHNUM/MIN"
                out_list.append("03")
                # "TECH"
                out_list.append(model_tech_code)
                # "TAU"
                out_list.append("{0:3d}".format(bt_storm.iloc[i]["TIME"]))
                # "LatN/S"
                lat_10 = int(round(bt_storm.iloc[i]["LAT"] * 10))
                lat_dir = "N" if lat_10 >= 0 else "S"
                out_list.append(f"{lat_10:3d}{lat_dir}")
                # LonE/W
                lon = bt_storm.iloc[i]["LON"]
                lon = lon + 360 if lon < 0 else lon
                lon_10 = int(round(lon * 10))
                lon_dir = "W" if lon >= 180 else "E"
                out_list.append(f"{lon_10:4d}{lon_dir}")
                # VMAX
                if i == 0:
                    vmax_curr += int(round(pred_storm.iloc[i][model_name]))
                else:
                    vmax_curr += int(round(pred_df.iloc[i][model_name] * 3 / 24))
                out_list.append(f"{vmax_curr:3d}")
                # MSLP
                out_list.append("    ")
                # TY
                out_list.append("XXX")
                # RAD
                out_list.append("   ")
                # WINDCODE
                out_list.append("AAA")
                # RAD1 to ROUTER
                out_list.extend(["    "] * 6)
                # RMOW to EYE
                out_list.extend(["   "] * 3)
                # SUBREGION
                basin_i = bt_storm.iloc[i]["BASIN"].upper()
                out_list.append(f"{basin_i:3s}")
                # MAXSEAS
                out_list.append("   ")
                # INITIALS
                out_list.append("DJG")
                # DIR
                dir = int(round(bt_storm.iloc[i]["STM_HDG"]))
                out_list.append(f"{dir:3d}")
                # SPEED
                speed = int(round(bt_storm.iloc[i]["STM_SPD"]))
                out_list.append(f"{speed:3d}")
                # STORMNAME
                s_name = bt_storm.iloc[i]["STNAM"].upper()
                out_list.append(f"{s_name:10s}")
                # DEPTH
                out_list.append("X")
                # SEAS
                out_list.append("  ")
                # SEASCODE
                out_list.append("   ")
                # SEAS1 to SEAS4
                out_list.extend(["    "] * 4)
                # VMAX DT
                out_list.append("VMAX_DT_24")
                pred_val = int(round(pred_storm.iloc[i][model_name]))
                out_list.append(f"{pred_val:3d}")
                if model_name + "_30" in pred_storm.columns:
                    ri_cols = [f"{model_name}_{x:02d}" for x in [30, 35, 40]]
                    ri_prob = int(round(pred_storm.iloc[i][ri_cols].values.sum() * 100))
                    out_list.append("P(dV/dt>30kt)")
                    out_list.append(f"{ri_prob:3d}")
                out_str = delimiter.join(out_list) + "\n"
                adeck_file.write(out_str)
    return

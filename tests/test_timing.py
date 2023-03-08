import sys
import argparse
import timeit

sys.path.append("..")

# internal
from assetra.core import (
    EnergySystemBuilder,
    StaticUnit,
    StochasticUnit,
    StorageUnit,
)
from assetra.probabilistic_analysis import ProbabilisticSimulation

# external
import xarray as xr
import numpy as np

ES = None
PS = None


def setup(
    num_static_units: int,
    num_stochastic_units: int,
    num_storage_units: int,
    num_hours: int,
):

    esb = EnergySystemBuilder()
    id_count = 0

    # add units
    for _ in range(num_static_units):
        esb.add_unit(
            StaticUnit(
                id=id_count,
                nameplate_capacity=-1,
                hourly_capacity=xr.DataArray(
                    data=np.ones(num_hours),
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00",
                            periods=num_hours,
                            freq="1H",
                        )
                    ),
                ),
            )
        )
        id_count += 1

    for _ in range(num_stochastic_units):
        esb.add_unit(
            StochasticUnit(
                id=id_count,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=np.ones(num_hours),
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00",
                            periods=num_hours,
                            freq="1H",
                        )
                    ),
                ),
                hourly_forced_outage_rate=xr.DataArray(
                    data=np.ones(num_hours) * 0.5,
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00",
                            periods=num_hours,
                            freq="1H",
                        )
                    ),
                ),
            )
        )
        id_count += 1

    for _ in range(num_storage_units):
        esb.add_unit(
            StorageUnit(
                id=id_count,
                nameplate_capacity=1,
                charge_rate=1,
                discharge_rate=1,
                charge_capacity=4,
                roundtrip_efficiency=0.8,
            )
        )
        id_count += 1

    global ES
    ES = esb.build()


def run(num_trials: int, num_hours: int):
    global ES
    global PS

    # setup simulation
    time_stamps = xr.date_range(
        start="2016-01-01 00:00:00", periods=num_hours, freq="1H"
    )
    start_time = time_stamps[0]
    end_time = time_stamps[-1]
    # time ps runtime
    ps = ProbabilisticSimulation(ES, start_time, end_time, num_trials)
    ps.run()


def test_assetra_timing(
    num_trials: int,
    num_static_units: int,
    num_stochastic_units: int,
    num_storage_units: int,
    num_hours: int,
    n: int = 1,
):

    print(
        "setup:",
        timeit.timeit(
            f"setup({num_static_units}, {num_stochastic_units}, {num_storage_units}, {num_hours})",
            number=1,
            globals=globals(),
        ),
    )

    print(
        "run:",
        timeit.timeit(
            f"run({num_trials}, {num_hours})", number=n, globals=globals()
        )
        / n,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("num_trials", type=int)
    parser.add_argument("num_static_units", type=int)
    parser.add_argument("num_stochastic_units", type=int)
    parser.add_argument("num_storage_units", type=int)
    parser.add_argument("num_hours", type=int)

    args = parser.parse_args()

    test_assetra_timing(
        args.num_trials,
        args.num_static_units,
        args.num_stochastic_units,
        args.num_storage_units,
        args.num_hours,
    )

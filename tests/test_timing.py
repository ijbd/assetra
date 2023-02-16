import sys
import argparse
from datetime import datetime, timedelta

sys.path.append("..")

# internal
from assetra.core import EnergySystem, StaticUnit, StochasticUnit, StorageUnit
from assetra.probabilistic_analysis import ProbabilisticSimulation
from assetra.adequacy_metrics import ExpectedUnservedEnergy

# external
import xarray as xr
import numpy as np


def test_assetra_timing(
    num_static_units: int,
    num_stochastic_units: int,
    num_storage_units: int,
    num_hours,
    num_trials: int,
):
    start_execution_time = datetime.now()

    # create system
    e = EnergySystem()
    id_count = 0

    # add units
    for _ in range(num_static_units):
        e.add_unit(
            StaticUnit(
                id=id_count,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=np.ones(num_hours),
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00", periods=num_hours, freq="1H"
                        )
                    ),
                ),
            )
        )
        id_count += 1

    for _ in range(num_stochastic_units):
        e.add_unit(
            StochasticUnit(
                id=id_count,
                nameplate_capacity=1,
                hourly_capacity=xr.DataArray(
                    data=np.ones(num_hours),
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00", periods=num_hours, freq="1H"
                        )
                    ),
                ),
                hourly_forced_outage_rate=xr.DataArray(
                    data=np.ones(num_hours) * 0.5,
                    coords=dict(
                        time=xr.date_range(
                            start="2016-01-01 00:00:00", periods=num_hours, freq="1H"
                        )
                    ),
                ),
            )
        )
        id_count += 1

    for _ in range(num_storage_units):
        e.add_unit(
            StorageUnit(
                id=id_count,
                charge_rate=1,
                discharge_rate=1,
                duration=4,
                roundtrip_efficiency=0.8,
            )
        )
        id_count += 1

    # setup simulation
    time_stamps = xr.date_range(
        start="2016-01-01 00:00:00", periods=num_hours, freq="1H"
    )
    start_time = time_stamps[0]
    end_time = time_stamps[-1]
    ps = ProbabilisticSimulation(e, start_time, end_time, num_trials)
    ps.run()

    print(f"--- execution: {datetime.now() - start_execution_time} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_static_units", type=int)
    parser.add_argument("num_stochastic_units", type=int)
    parser.add_argument("num_storage_units", type=int)
    parser.add_argument("num_hours", type=int)
    parser.add_argument("num_trials", type=int)

    args = parser.parse_args()

    test_assetra_timing(
        args.num_static_units,
        args.num_stochastic_units,
        args.num_storage_units,
        args.num_hours,
        args.num_trials,
    )

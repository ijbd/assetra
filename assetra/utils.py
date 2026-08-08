"""Convenience utilities for preparing input data.

This module collects small helper functions that make it easier to build the
data structures expected by :mod:`assetra.units`. They are not part of the
resource adequacy model itself; every function here is a shortcut for
formatting input data that users would otherwise write by hand.

Energy units accept hourly profiles as one-dimensional :class:`xarray.DataArray`
objects with a ``time`` dimension and datetime coordinates. Where a dataset
already carries its own timestamps, it is usually clearest to construct that
array directly. These helpers are aimed at the other case, where the data is a
plain sequence of hourly values that needs a datetime index attached to it, as
is common in small examples and prototypes.
"""

from datetime import datetime

import numpy as np
import xarray as xr


def get_hourly_time_series_xr(
    hourly_data: list[float],
    start_hour: str | datetime = "2019-01-01 00:00:00",
) -> xr.DataArray:
    """Return a formatted xarray data array for a sequence of hourly datapoints.

    Datapoints are assumed to be consecutive and evenly spaced at one-hour
    intervals, with the first datapoint corresponding to ``start_hour``. The
    resulting array is suitable for use as the hourly capacity or hourly forced
    outage rate profile of an energy unit.

    Args:
        hourly_data (list[float]): Input data stored as consecutive hour-scale
            datapoints.
        start_hour (str | datetime, optional): Timestamp corresponding to the
            first datapoint. Defaults to "2019-01-01 00:00:00".

    Returns:
        xr.DataArray: One-dimensional array with a time dimension and hourly
            datetime coordinates.

    Raises:
        ValueError: hourly_data contains no datapoints.

    Example:
        >>> hourly_demand = get_hourly_time_series_xr([100.0] * 8760)
        >>> hourly_demand.sizes["time"]
        8760
    """
    data = np.asarray(hourly_data, dtype=float)

    if data.size == 0:
        raise ValueError("hourly_data must contain at least one datapoint.")

    # cast to nanosecond precision to match the resolution xarray stores
    # internally, avoiding a conversion warning on construction
    time = np.asarray(
        xr.date_range(start_hour, freq="1h", periods=data.size),
        dtype="datetime64[ns]",
    )

    return xr.DataArray(
        data=data,
        dims=["time"],
        coords=dict(time=time),
    )
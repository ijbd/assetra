from datetime import datetime
from pathlib import Path
from logging import getLogger

# external libraries
from netCDF4 import Dataset
import pandas as pd

logger = getLogger(__name__)

EIA_860_NON_THERMAL_TECHNOLOGY = [
    "Onshore Wind Turbine",
    "Conventional Hydroelectric",
    "Solar Photovoltaic",
    "Offshore Wind Turbine",
    "Batteries",
    "Hydroelectric Pumped Storage",
]


class DataLoader:
    """This is a class interface for loading necessary data into assetra.
    It can be modified according to data availability.
    It should convert raw datasets into source-agnostic classes

    :param region_name: The name of the balancing authority for which this
        object will load data. Valid balancing authorities can be found @
        https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Data/blob/master/data/balancing_authority_acronyms.csv
    :type region_name: str
    :param year: The year for which data should be loaded
    :type year: int
    :param data_directory: The absolute path to the project's raw data directory.
        See the documentation for the expected structure of this directory
    :type data_directory: str
    """

    def __init__(self, region_name, year, data_directory):
        """Constructor method"""
        self.region_name = region_name
        self.year = year
        self.data_directory = data_directory

    def __enter__(self):
        self._load_eia_930_cleaned_hourly_demand()
        self._load_merra_temperature()
        self._load_merra_power_generation()
        self._load_eia_860_data()

    def __exit__(self):
        pass

    def _load_eia_930_cleaned_demand(self):
        """Load hourly demand data. This function expects a formatted csv
        named `data/eia930/<region_name>.csv`

        Data can be downloaded from:
        https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Data
        """
        logger.info("Loading hourly demand from cleaned EIA-930 dataset")
        # find file
        eia_930_file = Path(
            self.data_directory, "eia930", f"{self.region_name.upper()}.csv"
        )

        # open file
        try:
            # read
            eia_930_df = pd.read_csv(
                eia_930_file,
                usecols=["date_time", "cleaned demand (MW)"],
                index_col="date_time",
                parse_dates=True,
            )

            # subset by year
            eia_930_df = self._subset_df_by_year(eia_930_df)

            # remove leap day
            eia_930_df = self._remove_leap_day(eia_930_df)

            # keep demand
            self.hourly_demand = eia_930_df["cleaned demand (MW)"].values

        except FileNotFoundError as e:
            logger.error(f"Expected EIA-930 file located at: {eia_930_file}")
            raise e

    def _subset_df_by_year(self, df):
        """Select a single year from a date-time-
        indexed dataframe

        :param df: Pandas dataframe to subset
        :type df: class:`pandas.DataFrame`
        :return: Subset of `df` only including the dates only within
            `self.year`
        :rtype: class:`pandas.DataFrame`
        """
        return df[df.index.year == self.year]

    def _remove_leap_day(self, df):
        """Remove leap day from a date-time-indexed
        dataframe

        :param df: Pandas dataframe to subset
        :type df: class:`pandas.DataFrame`
        :return: Subset of `df` with any prior leap days removed
        :rtype: class:`pandas.DataFrame`
        """
        return df[(df.index.month != 2) | (df.index.day != 29)]

    def _load_merra_temperature(self):
        """This function loads hourly temperature from a combined
        MERRA NetCDF file. Input files
        can be created from https://github.com/ijbd/merra-power-generation
        """
        logger.info("Loading merra power generation data")
        # find file
        power_generation_file = Path(
            self.data_directory, "merra", f"merra_power_genration_{self.year}.nc"
        )

        # open file
        try:
            # read
            with Dataset(power_generation_file) as power_generation:
                self._pg_lats = power_generation["lat"][:]
                self._pg_lons = power_generation["lon"][:]
                self._pg_solar_cf = power_generation["solar_capacity_factor"][:]
                self._pg_wind_cf = power_generation["wind_capacity_factor"][:]

        except FileNotFoundError as e:
            logger.error(
                f"Expected merra power generation file located at: {power_generation_file}"
            )
            raise e

    def _load_merra_power_generation(self):
        """This function loads hourly solar and wind capacity factors
        from a merra-power-generation-generated NetCDF file. Input files
        can be created from https://github.com/ijbd/merra-power-generation
        """
        logger.info("Loading merra power generation data")
        # find file
        power_generation_file = Path(
            self.data_directory,
            "power_generation",
            f"merra_power_genration_{self.year}.nc",
        )

        # open file
        try:
            # read
            with Dataset(power_generation_file) as power_generation:
                self._pg_lats = power_generation["lat"][:]
                self._pg_lons = power_generation["lon"][:]
                self._pg_solar_cf = power_generation["solar_capacity_factor"][:]
                self._pg_wind_cf = power_generation["wind_capacity_factor"][:]

        except FileNotFoundError as e:
            logger.error(
                f"Expected merra power generation file located at: {power_generation_file}"
            )
            raise e

    def _load_eia_860_data(self):
        """This function loads all data needed *from* EIA 860
        dataset to populate the Fleet objects. However, additional
        data is needed.
        """
        self._load_eia_860_plants()
        self._load_eia_860_thermal()
        self._load_eia_860_solar()
        self._load_eia_860_wind()
        self._load_eia_860_storage()

    def _load_eia_860_plants(self):
        """This function loads the eia 860 plant data for the selected year and region"""
        logger.info("Loading plant data from EIA-860 dataset")
        # find file
        eia_860_plant_file = Path(
            self.data_directory, "eia860", f"2___Plant_Y{self.year}.xlsx"
        )

        # open file
        try:
            # read
            eia_860_plant_df = pd.read_excel(
                eia_860_plant_file,
                skiprows=1,
                usecols=[
                    "Plant Code",
                    "Latitude",
                    "Longitude",
                    "Balancing Authority Code",
                ],
                index_col="Plant Code",
            )

            # filter
            eia_860_plant_df = eia_860_plant_df[
                eia_860_plant_df["Balancing Authority Code"] == self.region_name
            ]

            self._plant_codes = eia_860_plant_df.index
            self._plant_latitudes = eia_860_plant_df["Latitude"]
            self._plant_longitudes = eia_860_plant_df["Longitude"]

        except FileNotFoundError as e:
            logger.error(
                f"Expected EIA-860 plant file located at: {eia_860_plant_file}"
            )
            raise e

    def _load_eia_860_thermal(self):
        """This function loads thermal generator data from eia 860 using plant
        data from _load_eia_860_plants. This function must be run after
        _load_eia_860_plants"""
        logger.info("Loading thermal generator data from EIA-860 dataset")
        # find file
        eia_860_generator_file = Path(
            self.data_directory, "eia860", f"3_1_Generator_Y{self.year}.xlsx"
        )

        # open file
        try:
            # read
            eia_860_generator_df = pd.read_excel(
                eia_860_generator_file,
                skiprows=1,
                usecols=[
                    "Plant Code",
                    "Technology",
                    "Nameplate Capacity (MW)",
                    "Status",
                ],
            )

            # filter by plants
            eia_860_generator_df = eia_860_generator_df[
                eia_860_generator_df["Plant Code"].isin(self._plant_codes)
            ]

            # filter by technology
            eia_860_generator_df = eia_860_generator_df[
                ~eia_860_generator_df["Technology"].isin(EIA_860_NON_THERMAL_TECHNOLOGY)
            ]

            # filter by status
            eia_860_generator_df = eia_860_generator_df[
                eia_860_generator_df["Status"] == "OP"
            ]

            # save
            self.thermal_capacity = eia_860_generator_df["Nameplate Capacity (MW)"]
            self.thermal_technology = eia_860_generator_df["Technology"]

            # get latitude array
            self.thermal_latitude = self._plant_latitudes[
                eia_860_generator_df["Plant Code"]
            ]
            self.thermal_longitude = self._plant_longitudes[
                eia_860_generator_df["Plant Code"]
            ]

        except FileNotFoundError as e:
            logger.error(
                f"Expected EIA-860 generator file located at: {eia_860_generator_file}"
            )
            raise e

    def _load_eia_860_wind(self):
        """This function loads wind generator data from eia 860 using plant
        data from _load_eia_860_plants. This function must be run after
        _load_eia_860_plants"""
        logger.info("Loading wind generator data from EIA-860 dataset")
        # find file
        eia_860_wind_file = Path(
            self.data_directory, "eia860", f"3_2_Wind_Y{self.year}.xlsx"
        )

        # open file
        try:
            # read
            eia_860_wind_df = pd.read_excel(
                eia_860_wind_file,
                skiprows=1,
                usecols=["Plant Code", "Nameplate Capacity (MW)", "Status"],
            )

            # filter by plants
            eia_860_wind_df = eia_860_wind_df[
                eia_860_wind_df["Plant Code"].isin(self._plant_codes)
            ]

            # filter by status
            eia_860_wind_df = eia_860_wind_df[eia_860_wind_df["Status"] == "OP"]

            # save
            self.wind_capacity = eia_860_wind_df["Nameplate Capacity (MW)"]

            # get latitude array
            self.wind_latitude = self._plant_latitudes[
                eia_860_wind_df["Plant Code"]
            ].values
            self.wind_longitude = self._plant_longitudes[
                eia_860_wind_df["Plant Code"]
            ].values

        except FileNotFoundError as e:
            logger.error(f"Expected EIA-860 wind file located at: {eia_860_wind_file}")
            raise e

    def _load_eia_860_solar(self):
        """This function loads solar generator data from eia 860 using plant
        data from _load_eia_860_plants. This function must be run after
        _load_eia_860_plants"""
        logger.info("Loading solar generator data from EIA-860 dataset")
        # find file
        eia_860_solar_file = Path(
            self.data_directory, "eia860", f"3_3_Solar_Y{self.year}.xlsx"
        )

        # open file
        try:
            # read
            eia_860_solar_df = pd.read_excel(
                eia_860_solar_file,
                skiprows=1,
                usecols=["Plant Code", "Nameplate Capacity (MW)", "Status"],
            )

            # filter by plants
            eia_860_solar_df = eia_860_solar_df[
                eia_860_solar_df["Plant Code"].isin(self._plant_codes)
            ]

            # filter by status
            eia_860_solar_df = eia_860_solar_df[eia_860_solar_df["Status"] == "OP"]

            # save
            self.solar_capacity = eia_860_solar_df["Nameplate Capacity (MW)"].values

            # get latitude array
            self.solar_latitude = self._plant_latitudes[
                eia_860_solar_df["Plant Code"]
            ].values
            self.solar_longitude = self._plant_longitudes[
                eia_860_solar_df["Plant Code"]
            ].values

        except FileNotFoundError as e:
            logger.error(
                f"Expected EIA-860 solar file located at: {eia_860_solar_file}"
            )
            raise e

    def _load_eia_860_storage(self):
        """This function loads storage generator data from eia 860 using plant
        data from _load_eia_860_plants. This function must be run after
        _load_eia_860_plants"""
        logger.info("Loading storage generator data from EIA-860 dataset")
        # find file
        eia_860_storage_file = Path(
            self.data_directory, "eia860", f"3_4_Energy_Storage_Y{self.year}.xlsx"
        )

        # open file
        try:
            # read
            eia_860_storage_df = pd.read_excel(
                eia_860_storage_file,
                skiprows=1,
                usecols=[
                    "Plant Code",
                    "Nameplate Capacity (MW)",
                    "Nameplate Energy Capacity (MWh)",
                    "Status",
                ],
            )

            # filter by plants
            eia_860_storage_df = eia_860_storage_df[
                eia_860_storage_df["Plant Code"].isin(self._plant_codes)
            ]

            # filter by status
            eia_860_storage_df = eia_860_storage_df[
                eia_860_storage_df["Status"] == "OP"
            ]

            # save
            self.storage_capacity = eia_860_storage_df["Nameplate Capacity (MW)"]
            self.storage_energy_capacity = eia_860_storage_df[
                "Nameplate Energy Capacity (MWh)"
            ]

            # get latitude array
            self.storage_latitude = self._plant_latitudes[
                eia_860_storage_df["Plant Code"]
            ].values
            self.storage_longitude = self._plant_longitudes[
                eia_860_storage_df["Plant Code"]
            ].values

        except FileNotFoundError as e:
            logger.error(
                f"Expected EIA-860 storage file located at: {eia_860_storage_file}"
            )
            raise e

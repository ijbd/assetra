from datetime import datetime
from pathlib import Path
from logging import getLogger
import pandas as pd

logger = getLogger(__name__)

class DataLoader:
	"""This is a class interface for loading necessary data into the 
	power system reliability model. It can be modified according to 
	data availability. It should convert raw datasets into source-agnostic 
	classes

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
		"""Constructor method
		"""
		self.region_name = region_name 
		self.year = year

	def __enter__(self):
		self._load_eia_930_cleaned_hourly_demand()
		self._load_merra_temperature()
		self._load_merra_power_generation()
		self._load_eia_860_fleet()

	def __exit__(self):
		pass

	def _load_merra_power_generation(self):
		pass

	def _load_eia_930_cleaned_demand(self):
		"""This function loads hourly demand data. It expects a formatted csv
		named `data/eia930/<region_name>.csv`
		
		Data can be downloaded from:
		https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Data
		"""
		logger.info("Loading hourly demand from cleaned EIA-930 dataset")
		# find file
		eia_930_file = Path(self.data_directory, 'eia930', f'{self.region_name.upper()}.csv')

		# open file
		eia_930_index_col = 'date_time'
		eia_930_demand_col = 'cleaned demand (MW)'
		try:
			# read 
			eia_930_df = pd.read_csv(
				eia_930_file, 
				index_col=eia_930_index_col, 
				parse_dates=True
			)

			# subset by year
			eia_930_df = self._subset_by_year(eia_930_df)

			# remove leap day
			eia_930_df = self._remove_leap_day(eia_930_df)

			# keep demand
			self.demand = eia_930_df[eia_930_demand_col]
			
		except FileNotFoundError as e:
			logger.error(f'Expected EIA-930 file located at: {eia_930_file}')
			raise e
		
	def _subset_df_by_year(self, df):
		"""This function selects a single year from a date-time-
		indexed dataframe

		:param df: Pandas dataframe to subset
		:type df: class:`pandas.DataFrame`
		:return: Subset of `df` only including the dates only within
			`self.year`
		:rtype: class:`pandas.DataFrame`
		"""
		return df.between_time(
				datetime(self.year,1,1),
				datetime(self.year+1,1,1),
				include_start=True,
				include_end=False
			)

	def _remove_leap_day(self, df):
		"""This function removes leap day from a date-time-indexed
		dataframe

		:param df: Pandas dataframe to subset
		:type df: class:`pandas.DataFrame`
		:return: Subset of `df` with any prior leap days removed
		:rtype: class:`pandas.DataFrame`
		"""
		return df[~(df.index.month == 2 & df.index.day == 29)]

	def _load_merra_temperature(self):
		

	def _load_eia_860_fleet(self):
		pass

	def _load_eia_860_plants(self):
		pass

	def _load_eia_860_thermal(self):
		pass

	def _load_eia_860_solar(self):
		pass

	def _load_eia_860_wind(self):
		pass

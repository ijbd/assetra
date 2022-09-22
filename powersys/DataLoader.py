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
		self.data_directory = data_directory

	def __enter__(self):
		self._load_eia_930_cleaned_hourly_demand()
		self._load_merra_temperature()
		self._load_merra_power_generation()
		self._load_eia_860_fleet()

	def __exit__(self):
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
		try:
			# read 
			eia_930_df = pd.read_csv(
				eia_930_file, 
				usecols=[
					'date_time',
					'cleaned demand (MW)'
				],
				index_col='date_time', 
				parse_dates=True
			)

			# subset by year
			eia_930_df = self._subset_df_by_year(eia_930_df)

			# remove leap day
			eia_930_df = self._remove_leap_day(eia_930_df)

			# keep demand
			subset = eia_930_df['cleaned demand (MW)']

			self.time_stamps = subset.index.values
			self.hourly_demand = subset.values
			
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
		return df[df.index.year == self.year]

	def _remove_leap_day(self, df):
		"""This function removes leap day from a date-time-indexed
		dataframe

		:param df: Pandas dataframe to subset
		:type df: class:`pandas.DataFrame`
		:return: Subset of `df` with any prior leap days removed
		:rtype: class:`pandas.DataFrame`
		"""
		return df[(df.index.month != 2) | (df.index.day != 29)]

	def _load_merra_temperature(self):
		"""This function loads hourly 2-meter temperature data from
		a MERRA netcdf file.
		"""
		logger.info("Loading hourly demand from MERRA dataset")
		# find file

		# open file
		
	def _load_merra_power_generation(self):
		"""This function loads hourly solar and wind capacity factors
		from a merra-power-generation-generated NetCDF file. Input files
		can be created from https://github.com/ijbd/merra-power-generation
		"""
		pass

	def _load_eia_860_fleet(self):
		"""This function loads all data needed *from* EIA 860
		dataset to populate the Fleet objects. However, additional
		data is needed. 
		"""
		self._load_eia_860_plants()
		self._load_eia_860_generators()
		self._load_eia_860_solar()
		self._load_eia_860_wind()


	def _load_eia_860_plants(self):
		"""This function loads the eia 860 plant data"""
		logger.info("Loading hourly demand from cleaned EIA-930 dataset")
		# find file
		eia_860_plant_file = Path(self.data_directory, 'eia860', f'2___Plant_Y{self.year}.xlsx')

		# open file
		try:
			# read 
			eia_860_plant_df = pd.read_excel(
				eia_860_plant_file,
				skiprows=1,
				usecols=[
					'Plant Code',
					'Latitude',
					'Longitude',
					'Balancing Authority Code'
				],
				index_col='Plant Code',
			)

			# filter
			eia_860_plant_df = eia_860_plant_df[
				eia_860_plant_df['Balancing Authority Code'] == self.region_name
			]

			self._plant_codes = eia_860_plant_df.index
			self._plant_latitudes = eia_860_plant_df['Latitude']
			self._plant_longitudes = eia_860_plant_df['Longitude']
				
		except FileNotFoundError as e:
			logger.error(f'Expected EIA-860 plant file located at: {eia_860_plant_file}')
			raise e


	def _load_eia_860_thermal(self):
		pass

	def _load_eia_860_solar(self):
		pass

	def _load_eia_860_wind(self):
		pass

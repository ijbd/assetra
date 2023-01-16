import unittest 
import sys
from pathlib import Path

sys.path.append('..')

from powersys.preprocessing import data_loader

class TestDataLoader(unittest.TestCase):
	def setUp(self):
		self.test_arguments = (
			'PACE', 
			2019, 
			Path('..','test_data')
		)

	def test_load_eia_930_cleaned_demand(self):
		d = data_loader.DataLoader(*self.test_arguments)
		d._load_eia_930_cleaned_demand()

		self.assertEqual(len(d.hourly_demand), 8760)

	def test_load_eia_860_plants(self):
		d = data_loader.DataLoader(*self.test_arguments)
		d._load_eia_860_plants()
		
		self.assertEqual(d._plant_codes[0], 159)
		self.assertEqual(len(d._plant_codes), len(d._plant_latitudes))
		self.assertEqual(len(d._plant_latitudes), len(d._plant_longitudes))

	def test_load_eia_860(self):
		d = data_loader.DataLoader(*self.test_arguments)
		d._load_eia_860_data()

		# first thermal generator
		self.assertAlmostEqual(d.thermal_capacity[0], 30.7)
		self.assertEqual(d.thermal_technology[0], 'Geothermal')
		self.assertAlmostEqual(d.thermal_latitude[0], 38.488900)
		self.assertAlmostEqual(d.thermal_longitude[0], -112.853300)

		# first solar generator
		self.assertAlmostEqual(d.solar_capacity[0], 0.8)
		self.assertAlmostEqual(d.solar_latitude[0], 40.767222)
		self.assertAlmostEqual(d.solar_longitude[0], -111.895000)

		# first wind generator
		self.assertAlmostEqual(d.wind_capacity[0], 0.8)
		self.assertAlmostEqual(d.wind_latitude[0], 40.767222)
		self.assertAlmostEqual(d.wind_longitude[0], -111.895000)

		# first storage generator


if __name__ == '__main__':
	unittest.main()
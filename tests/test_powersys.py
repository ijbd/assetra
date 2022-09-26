import unittest 
import sys
from pathlib import Path

sys.path.append('..')

from powersys import DataLoader

class TestDataLoader(unittest.TestCase):
	def setUp(self):
		self.test_arguments = (
			'PACE', 
			2019, 
			Path('..','test_data')
		)

	def test_load_eia_930_cleaned_demand(self):
		d = DataLoader.DataLoader(*self.test_arguments)
		d._load_eia_930_cleaned_demand()

		self.assertEqual(len(d.hourly_demand), 8760)

	def test_load_eia_860_plants(self):
		d = DataLoader.DataLoader(*self.test_arguments)
		d._load_eia_860_plants()
		
		self.assertEqual(len(d._plant_codes), len(d._plant_latitudes))
		self.assertEqual(len(d._plant_latitudes), len(d._plant_longitudes))

	def test_load_eia_860_thermal(self):
		d = DataLoader.DataLoader(*self.test_arguments)
		d._load_eia_860_plants()
		d._load_eia_860_thermal()

		thermal_fleet_size = len(d.thermal_capacity)
		
		self.assertEqual(len(d.thermal_technology), thermal_fleet_size)
		self.assertEqual(len(d.thermal_latitude), thermal_fleet_size)
		self.assertEqual(len(d.thermal_longitude), thermal_fleet_size)

	def test_context(self):
		pass

if __name__ == '__main__':
	unittest.main()
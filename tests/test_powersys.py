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

		self.assertEqual(len(d.time_stamps), 8760)
		self.assertEqual(len(d.hourly_demand), 8760)

	def test_load_eia_860_plants(self):
		d = DataLoader.DataLoader(*self.test_arguments)
		d._load_eia_860_plants()

		print(d)

	def test_context(self):
		pass

if __name__ == '__main__':
	from powersys import DataLoader
	unittest.main()
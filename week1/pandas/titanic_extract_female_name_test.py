import unittest

from titanic_extract_female_name import to_female_name


class MyTestCase(unittest.TestCase):
    def test_miss_name_extraction(self):
        extracted_name = to_female_name("Wick, Miss. Mary Natalie")
        self.assertEqual(extracted_name, 'Mary')

    def test_mrs_name_extraction(self):
        extracted_name = to_female_name("Turpin, Mrs. William John Robert (Dorothy Ann Wonnacott)")
        self.assertEqual(extracted_name, 'Dorothy')

    def test_mrs_no_brackets_extraction(self):
        extracted_name = to_female_name("Masselmani, Mrs. Fatima")
        self.assertEqual(extracted_name, 'Fatima')

    def test_mrs_doctor_extraction(self):
        extracted_name = to_female_name("Leader, Dr. Alice (Farnham)")
        self.assertEqual(extracted_name, 'Alice')

    def test_wrong_not_mrs_not_miss_name(self):
        with self.assertRaises(ValueError):
            to_female_name("some random value")


if __name__ == '__main__':
    unittest.main()

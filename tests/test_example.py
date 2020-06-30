"""test unittest"""


import unittest



class ExampleTest(unittest.TestCase):
    """Example test case"""

    def test_example(self):
        """example"""

        self.assertEqual(1, 1)



if __name__ == '__main__':
    unittest.main()

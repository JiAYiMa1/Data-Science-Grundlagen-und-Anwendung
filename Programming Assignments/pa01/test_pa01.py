import unittest
from pa01.pa01 import world_greeter
from pa01.pa01 import add_a_b
from pa01.pa01 import smallest_element


class TestGreetings(unittest.TestCase):
    def test_world_greeter(self):
        self.assertEqual(world_greeter(), "Hello World!")

class TestSum(unittest.TestCase):
    def test_add_a_b(self):
        self.assertEqual(add_a_b(1,1), 2)

class TestSmallestElement(unittest.TestCase):
    def test_smallest_element(self):
        test_list = [1,2,3,4,5]
        self.assertEqual(smallest_element(test_list), 1)


if __name__ == '__main__':
    unittest.main()

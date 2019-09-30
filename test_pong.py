import unittest

from pgpong import sigmoid

class TestPong(unittest.TestCase):
    
    def test_sigmoid(self):
        assert sigmoid(0) == 0.5



if __name__ == "__main__":
	unittest.main()

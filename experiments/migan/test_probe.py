"""Fast unit tests: no model downloads or ONNX Runtime loading."""
import unittest
import numpy as np
from PIL import Image
from probe import prepare, composite

class ProbeTests(unittest.TestCase):
    def test_polarity_and_chw(self):
        im = Image.fromarray(np.array([[[10,20,30],[40,50,60]]], np.uint8))
        mask = Image.fromarray(np.array([[0,255]], np.uint8))
        rgb, known, src, sel = prepare(im, mask)
        self.assertEqual(rgb.shape, (1,3,1,2))
        self.assertEqual(known.shape, (1,1,1,2))
        self.assertEqual(rgb[0,:,0,1].tolist(), [40,50,60])
        self.assertEqual(known[0,0,0].tolist(), [255,0])
        self.assertEqual(src.shape, (1,2,3))
        self.assertEqual(sel.tolist(), [[0,255]])

    def test_pixel_integrity_and_feathering(self):
        old = np.full((5,5,3), 90, np.uint8)
        new = np.full((5,5,3), 230, np.uint8)
        mask = np.zeros((5,5), np.uint8)
        mask[1:4,1:4] = 255
        mask[2,2] = 128
        result = composite(old, new, mask)
        np.testing.assert_array_equal(result[mask == 0], old[mask == 0])
        self.assertTrue(np.all(result[mask == 255] == 230))
        self.assertTrue(np.all((result[2,2] > 90) & (result[2,2] < 230)))

    def test_bad_input(self):
        im = Image.new("RGB", (3,4))
        for mask in (Image.new("L",(4,3)), Image.new("L",(3,4),0), Image.new("L",(3,4),255)):
            with self.assertRaises(ValueError):
                prepare(im, mask)
        old = np.zeros((2,2,3), np.uint8)
        with self.assertRaises(ValueError):
            composite(old, old.astype(np.float32), np.zeros((2,2), np.uint8))

if __name__ == "__main__":
    unittest.main()

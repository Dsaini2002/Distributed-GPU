import unittest
import torch
from studio_zoo.models import CATALOG, build_model


class ModelTests(unittest.TestCase):
    def test_registry(self):
        self.assertGreaterEqual(len(CATALOG), 15)

    def test_tabular(self):
        x = build_model('tabular_mlp_regression', num_inputs=3)
        self.assertEqual(tuple(x(torch.randn(2, 3)).shape), (2, 1))

    def test_image_cnn(self):
        x = build_model('image_cnn', num_classes=2)
        self.assertEqual(tuple(x(torch.randn(2, 3, 64, 64)).shape), (2, 2))

    def test_unet(self):
        x = build_model('unet_segmentation')
        self.assertEqual(tuple(x(torch.randn(2, 3, 64, 64)).shape), (2, 1, 64, 64))

    def test_tiny_vit(self):
        x = build_model('tiny_vit', num_classes=3)
        self.assertEqual(tuple(x(torch.randn(1, 3, 64, 64)).shape), (1, 3))

    def test_text_gru(self):
        x = build_model('text_gru', num_classes=4)
        self.assertEqual(tuple(x(torch.randint(1, 100, (2,128))).shape), (2,4))

    def test_text_transformer(self):
        x = build_model('text_transformer', num_classes=3)
        self.assertEqual(tuple(x(torch.randint(1, 100, (2,128))).shape), (2,3))

    def test_tiny_lm(self):
        x = build_model('tiny_transformer_lm', vocab_size=50)
        self.assertEqual(tuple(x(torch.randint(0, 50, (2,64))).shape), (2,64,50))

    def test_timeseries(self):
        x = build_model('time_series_lstm')
        self.assertEqual(tuple(x(torch.randn(2,32,1)).shape), (2,1))


if __name__ == '__main__':
    unittest.main()

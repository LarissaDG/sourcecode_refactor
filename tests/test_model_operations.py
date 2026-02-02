import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.utils.model_operations import process_images


class TestModelOperations(unittest.TestCase):

    @patch("src.utils.model_operations.generate_description")
    def test_process_images(self, mock_generate):
        mock_generate.return_value = "description"

        df = pd.DataFrame({
            "filename": ["img1.png", "img2.png"]
        })

        vl_chat_processor = MagicMock()
        tokenizer = MagicMock()
        vl_gpt = MagicMock()

        out = process_images(df, "/tmp", vl_chat_processor, tokenizer, vl_gpt)

        self.assertEqual(out["Description"].tolist(), ["description", "description"])
        self.assertEqual(mock_generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()

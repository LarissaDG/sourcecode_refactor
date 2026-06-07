import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# -------------------------------------------------
# MOCK DOS MÓDULOS PESADOS (janus, transformers etc.)
# -------------------------------------------------
sys.modules["janus"] = MagicMock()
sys.modules["janus.models"] = MagicMock()
sys.modules["janus.utils"] = MagicMock()
sys.modules["janus.utils.io"] = MagicMock()

# Agora o import funciona sem erro
from src.image_to_text.imgTex_operations import process_images


class TestImgTexOperations(unittest.TestCase):

    @patch("src.image_to_text.imgTex_operations.generate_description")
    def test_process_images(self, mock_generate):
        # Arrange
        mock_generate.return_value = "description"

        df = pd.DataFrame({
            "filename": ["img1.png", "img2.png"]
        })

        vl_chat_processor = MagicMock()
        tokenizer = MagicMock()
        vl_gpt = MagicMock()

        # Act
        out = process_images(
            df=df,
            image_base_path="/tmp",
            vl_chat_processor=vl_chat_processor,
            tokenizer=tokenizer,
            vl_gpt=vl_gpt
        )

        # Assert
        self.assertEqual(out["Description"].tolist(), ["description", "description"])
        self.assertEqual(mock_generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()

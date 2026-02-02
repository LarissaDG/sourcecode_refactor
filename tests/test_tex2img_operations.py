import unittest
import os
import torch
import numpy as np
import pandas as pd
import PIL.Image

from unittest.mock import MagicMock, patch

from src.text_to_image.texImg_operations import get_prompt, run_model_generationc


class TestGenerationOperations(unittest.TestCase):

    def setUp(self):
        # Cria um dataframe fake
        self.df = pd.DataFrame({
            "Description_en": ["A cat", "A dog"]
        })

        self.output_dir = "temp_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Cria imagem fake para gerar tokens (mock)
        self.fake_image = PIL.Image.new("RGB", (32, 32), color="red")

    def tearDown(self):
        # Remove arquivos criados
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)

    def test_get_prompt_alternate(self):
        processor = MagicMock()
        processor.apply_sft_template_for_multi_turn_prompts.return_value = "PROMPT"
        processor.image_start_tag = "<img>"
        processor.image_gen_tag = "<imggen>"

        prompt = get_prompt(processor, "Hello", is_token_based=True, use_alternate=True)

        self.assertIn("PROMPT", prompt)
        self.assertTrue(prompt.endswith(processor.image_start_tag))

    @patch("generation_operations.generate_token_based")
    @patch("generation_operations.VLChatProcessor")
    @patch("generation_operations.AutoModelForCausalLM")
    def test_run_model_generation(self, mock_model, mock_processor, mock_generate_token):
        # Mock do processor e do modelo
        processor_instance = MagicMock()
        processor_instance.tokenizer.encode.return_value = [1, 2, 3]
        processor_instance.apply_sft_template_for_multi_turn_prompts.return_value = "PROMPT"
        processor_instance.image_start_tag = "<img>"
        processor_instance.image_gen_tag = "<imggen>"
        processor_instance.pad_id = 0

        model_instance = MagicMock()
        model_instance.to.return_value = model_instance
        model_instance.eval.return_value = model_instance

        mock_processor.from_pretrained.return_value = processor_instance
        mock_model.from_pretrained.return_value = model_instance

        run_model_generation(
            model_path="fake/path",
            df=self.df,
            output_dir=self.output_dir,
            use_alternate=True,
            model_name_prefix="TEST"
        )

        # Verifica se gerou os arquivos CSV
        self.assertTrue(os.path.exists("sampled_TEST_with_gen.csv"))

        # Remove CSV
        os.remove("sampled_TEST_with_gen.csv")

        # Verifica se a função generate_token_based foi chamada
        self.assertTrue(mock_generate_token.called)


if __name__ == "__main__":
    unittest.main()

from unittest import mock
from unittest.mock import ANY

from flash_gradio import FlashGradio


@mock.patch("flash_gradio.component.gradio")
def test_flash_gradio_text_classification(gradio_mock):
    # Sample run data config to test workflow
    run_dict = {
        "task": "text_classification",
        "checkpoint_path": "checkpoint.ckpt",
    }

    flash_gradio = FlashGradio()
    flash_gradio.run(
        run_dict["task"],
        run_dict["checkpoint_path"],
    )
    gradio_mock.Interface.assert_called_once_with(fn=ANY, inputs=ANY, outputs=ANY)

<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A Lightning component to serve a Flash Task using Gradio

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/PyTorchLightning/LAI-flash-gradio.git
cd LAI-flash-gradio
pip install -r requirements.txt
pip install -e .
```

## Use the component

Copy the following code to a file `app.py`, and run the app using: `lightning run app app.py`.

```python
import lightning as L
from lightning import LightningApp

from flash_gradio import FlashGradio


class FlashGradioComponent(L.LightningFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flash_gradio = FlashGradio()

    def run(self):
        # Pass a checkpoint path for Gradio to load
        checkpoint_path = "checkpoint.ckpt"

        run_dict = {
            "task": "text_classification",
            "url": "https://pl-flash-data.s3.amazonaws.com/imdb.zip",
            "data_config": {
                "target": "from_csv",
                "input_field": "review",
                "target_fields": "sentiment",
                "train_file": "imdb/train.csv",
                "val_file": "imdb/valid.csv",
            },
        }

        self.flash_gradio.run(
            run_dict["task"],
            run_dict["url"],
            run_dict["data_config"],
            checkpoint_path,
        )

    def configure_layout(self):
        layout = [
            {
                "name": "Predictions Explorer (Gradio)",
                "content": self.flash_gradio,
            },
        ]
        return layout


# To launch the gradio component
app = LightningApp(FlashGradioComponent(), debug=True)
```

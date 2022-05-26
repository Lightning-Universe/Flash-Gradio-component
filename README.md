<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A Lightning component to serve a Flash Task using Gradio

______________________________________________________________________

</div>

## Use the component

Note that we have a `run_once` argument to the component, this allows you to only run it once if needed. Default is `True`, which means it will only run once if not set `False` explicitly.

```python
import lightning as L
from lightning import LightningApp

from flash_gradio import FlashGradio


class FlashGradioComponent(L.LightningFlow):
    def __init__(self):
        super().__init__()
        # We only run FlashGradio once, since we only have one input
        # default for `run_once` is `True` as well
        self.flash_gradio = FlashGradio(run_once=True)
        self.layout = []

    def run(self):
        # Pass a checkpoint path for Gradio to load
        checkpoint_path = "checkpoint.pt"

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
        if self.flash_gradio.ready and not self.layout:
            self.layout.append(
                {
                    "name": "Predictions Explorer (Gradio)",
                    "content": self.flash_gradio,
                },
            )
        return self.layout


# To launch the gradio component
app = LightningApp(FlashGradioComponent(), debug=True)
```

## Install

Use these instructions to install:

```bash
git clone https://github.com/PyTorchLightning/LAI-flash-gradio.git
cd LAI-flash-gradio
pip install -r requirements.txt
pip install -e .
```

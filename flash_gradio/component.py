import logging
import os
import tempfile
from typing import Dict, Optional
from types import ModuleType
import shutil

from lightning.utilities.imports import _is_gradio_available

if _is_gradio_available():
    import gradio
else:
    gradio = ModuleType("gradio")

from flash_gradio import tasks
from flash_gradio.utilities import generate_script
from lightning.components.python import TracerPythonScript
from lightning.storage.path import Path


class FlashGradio(TracerPythonScript):
    def __init__(self, *args, parallel=True, run_once=False, **kwargs):
        super().__init__(
            __file__,
            *args,
            parallel=parallel,
            run_once=run_once,
            **kwargs
        )

        self.script_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(self.script_dir, "flash_gradio.py")
        self.script_options = {"task": None, "data_config": None, "url": None}
        self._task_meta: Optional[tasks.TaskMeta] = None
        self.checkpoint_path = None
        self.enable_queue = False

        self.sample_input = "Lightning rocks!"

    def run(self, task: str, url: str, data_config: Dict, checkpoint: Path):
        self._task_meta = getattr(tasks, task)

        self.checkpoint_path = checkpoint
        self.script_options["task"] = task
        self.script_options["data_config"] = data_config
        self.script_options["url"] = url

        interface = gradio.Interface(
            fn=self.predict,
            inputs=[
                gradio.inputs.Textbox(default=self.sample_input, label="Input"),
            ],
            outputs=[
                gradio.outputs.Textbox(type="text", label="Output")
            ],
        )
        logging.info(f"Launching Gradio server at: {self.url}")
        interface.launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )

    def predict(self, text):
        generate_script(
            self.script_path,
            "flash_gradio.jinja",
            task=self.script_options["task"],
            data_module_import_path=self._task_meta.data_module_import_path,
            data_module_class=self._task_meta.data_module_class,
            task_import_path=self._task_meta.task_import_path,
            task_class=self._task_meta.task_class,
            url=self.url,
            data_config=self.script_options["data_config"],
            checkpoint_path=str(self.checkpoint_path),
            input_text=str(text),
        )
        self.on_before_run()
        init_globals = globals()
        env_copy = os.environ.copy()
        if self.env:
            os.environ.update(self.env)
        res = self._run_tracer(init_globals)
        os.environ = env_copy
        return res["predictions"][0][0]

    def on_exit(self):
        shutil.rmtree(self.script_dir)

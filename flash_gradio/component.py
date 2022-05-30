import logging
import os
import tempfile
from typing import Dict, Optional
import threading

import gradio as gr
from flashy.components import tasks
from flashy.components.tasks import TaskMeta
from flashy.components.utilities import generate_script
from lightning.components.python import TracerPythonScript
from lightning.storage.path import Path


class FlashGradio(TracerPythonScript):
    def __init__(self, *args, parallel=True, run_once=True, **kwargs):
        super().__init__(__file__, *args, parallel=parallel, run_once=run_once, **kwargs)

        self.script_dir = tempfile.mkdtemp()
        self.script_path = os.path.join(self.script_dir, "flash_gradio.py")
        self.script_options = {"task": None, "data_config": None, "url": None}
        self._task_meta: Optional[TaskMeta] = None
        self._checkpoint = None

        self.ready = False

    def run(self, task: str, url: str, data_config: Dict, checkpoint: Path):
        self.ready = False
        self._task_meta = getattr(tasks, task)

        # This is bad, we should not do this.
        self._checkpoint = checkpoint
        self.script_options["task"] = task
        self.script_options["data_config"] = data_config
        self.script_options["url"] = url

        # We don't call super().run() as we don't want to run the tracer yet!!
        # In case the definition of run method changes for the
        # `TracerPythonScript`,
        # this part of code will have to be changed. Bad workaround!

        # Why can't we run the tracer yet? The script expects a user input,
        # which will be available after the Gradio Interface is launched
        # (`demo.launch`). demo.launch blocks any command after it, and hence
        # super().run() will never be called if we put it here. The tracer
        # should run in the callback function `_apply` which gets the user
        # input text
        return self.on_after_run({})

    def launch_gradio(self, interface):
        interface.launch(
            server_name=self.host,
            server_port=self.port,
        )

    def on_after_run(self, res):
        logging.info("Launching Gradio server")

        sample_input = "Lightning rocks!"
        interface = gr.Interface(
            fn=self.predict,
            inputs=[
                gr.inputs.Textbox(default=sample_input, label="Input"),
            ],
            outputs=[
                gr.outputs.Textbox(type="text", label="Output")
            ],
        )

        th = threading.Thread(target=self.launch_gradio, args=(interface, ))
        th.start()
        if th.is_alive():
            self.ready = True

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
            checkpoint=str(self._checkpoint),
            input_text=str(text),
        )
        self.on_before_run()
        init_globals = globals()
        env_copy = os.environ.copy()
        if self.env:
            os.environ.update(self.env)
        res = self._run_tracer(init_globals)
        os.environ = env_copy
        return res["predictions"]

import os

import lumen.ai as lmai
import panel as pn

from lumen_anndata.ui import build_ui

pn.config.disconnect_notification = "Connection lost, try reloading the page!"
pn.config.ready_notification = "Application fully loaded."
pn.extension("filedropper", "jsoneditor")

llm = lmai.llm.OpenAI(
    api_key=os.environ["HF_API_TOKEN"],
    endpoint="https://router.huggingface.co/v1",
    model_kwargs={"default": {"model": "google/gemma-4-31B-it"}},
)

build_ui(llm=llm).servable()

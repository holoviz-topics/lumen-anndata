import panel as pn

from lumen_anndata.ui import build_ui

pn.config.disconnect_notification = "Connection lost, try reloading the page!"
pn.config.ready_notification = "Application fully loaded."
pn.extension("filedropper", "jsoneditor")

build_ui().servable()

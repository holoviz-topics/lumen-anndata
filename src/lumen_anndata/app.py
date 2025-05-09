from pathlib import Path

import lumen.ai as lmai
import panel as pn

pn.config.disconnect_notification = "Connection lost, try reloading the page!"
pn.config.ready_notification = "Application fully loaded."
pn.extension("filedropper")

INSTRUCTIONS = """
You are an expert scientist working in Python, with a specialty using Anndata and Scanpy.
Help the user with their questions, and if you don't know the answer, say so.
"""

db_uri = str(Path(__file__).parent / "embeddings" / "scanpy.db")
vector_store = lmai.vector_store.DuckDBVectorStore(uri=db_uri, embeddings=lmai.embeddings.OpenAIEmbeddings())
doc_lookup = lmai.tools.VectorLookupTool(vector_store=vector_store, n=3)

ui = lmai.ExplorerUI(
    agents=[lmai.agents.ChatAgent(tools=[doc_lookup], template_overrides={"main": {"instructions": INSTRUCTIONS}})], default_agents=[], log_level="debug"
)
ui.servable()

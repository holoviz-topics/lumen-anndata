import subprocess

subprocess.run(
    [
        "wget",
        "--mirror",
        "--convert-links",
        "--adjust-extension",
        "--no-parent",
        "--accept=html,htm",
        "https://scanpy.readthedocs.io/en/stable/",
    ]
)

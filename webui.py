from modules import ui, cmd_opts
import faiss


def webui():
    app = ui.create_ui()
    app, local_url, share_url = app.launch(
        server_name=cmd_opts.opts.host,
        server_port=cmd_opts.opts.port,
        share=cmd_opts.opts.share,
    )

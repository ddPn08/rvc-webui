from modules import ui, cmd_opts
import faiss


def webui():
    app = ui.create_ui()
    app, local_url, share_url = app.queue(concurrency_count=511, max_size=1022).launch(
        server_name=cmd_opts.opts.host,
        server_port=cmd_opts.opts.port,
        share=cmd_opts.opts.share,
    )

if __name__ == "__main__":
    webui()
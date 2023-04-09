import modules.ui as ui


def webui():
    app = ui.create_ui()
    app, local_url, share_url = app.launch()

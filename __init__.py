"""
dashboard/__init__.py

App factory for the small research-tool dashboard built on top of
Phases 1-3's pipeline. Nothing here computes anything itself — see
data_service.py for that; this module only wires Flask up.
"""

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    app.jinja_env.filters["usd"] = lambda v: f"${v:,.0f}" if v is not None else "n/a"
    app.jinja_env.filters["pct"] = lambda v: f"{v * 100:+.2f}%" if v is not None else "n/a"
    app.jinja_env.filters["datefmt"] = lambda d: d.strftime("%Y-%m-%d") if d is not None else "n/a"

    from dashboard.routes import bp
    app.register_blueprint(bp)

    return app

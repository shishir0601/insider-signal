"""
run_dashboard.py

Entrypoint for the Phase 4 dashboard. Run with:

    python3 run_dashboard.py

then open http://127.0.0.1:5050
"""

from dashboard import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)

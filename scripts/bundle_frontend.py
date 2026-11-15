#!/usr/bin/env python3
"""Empacota o frontend em um único arquivo HTML autônomo (`frontend/dist/index.html`).

Permite implantar a interface 3D em qualquer host estático (S3, GitHub Pages, Netlify,
Cloudflare Pages, Nginx) apontando para um backend MuJoCo remoto via `?ws=ws://IP:8000/ws`.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DIST_FILE = FRONTEND_DIR / "dist" / "index.html"


def build_single_file_frontend() -> Path:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css = (FRONTEND_DIR / "css" / "style.css").read_text(encoding="utf-8")
    ws_js = (FRONTEND_DIR / "js" / "websocket_client.js").read_text(encoding="utf-8")
    renderer_js = (FRONTEND_DIR / "js" / "renderer.js").read_text(encoding="utf-8")
    controls_js = (FRONTEND_DIR / "js" / "controls.js").read_text(encoding="utf-8")

    html = html.replace(
        '<link rel="stylesheet" href="css/style.css" />',
        f"<style>\n{css}\n</style>",
    )
    html = html.replace(
        '<script src="js/websocket_client.js"></script>',
        f"<script>\n{ws_js}\n</script>",
    )
    html = html.replace(
        '<script src="js/renderer.js"></script>',
        f"<script>\n{renderer_js}\n</script>",
    )
    html = html.replace(
        '<script src="js/controls.js"></script>',
        f"<script>\n{controls_js}\n</script>",
    )
    # dental_scene.js é ES module (Three.js CDN) — mantém referência relativa;
    # no standalone, o backend ainda precisa servir /js/dental_scene.js + /assets/.
    # (Não inline: depende de importmap + three/addons.)

    DIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIST_FILE.write_text(html, encoding="utf-8")
    return DIST_FILE


if __name__ == "__main__":
    out = build_single_file_frontend()
    print(f"Bundle single-file gerado em: {out} ({out.stat().st_size} bytes)")

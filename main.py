"""
main.py
API de análisis de imágenes — sin dependencias de Google/Gemini.
Endpoints:
  POST /analizar          → analiza imagen subida como archivo
  POST /analizar-url      → analiza imagen desde URL externa
  GET  /health            → verifica que la API está viva
"""

import io
import urllib.request
import urllib.error
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.analyzer import analizar_imagen_completa

app = FastAPI(
    title="Image Analyzer API",
    description="Análisis profundo de imágenes: colores, figuras, personajes, composición.",
    version="1.0.0",
)

# Permitir llamadas desde cualquier origen (InfinityFree, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MIMES_PERMITIDOS = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_BYTES = 15 * 1024 * 1024  # 15 MB


class UrlRequest(BaseModel):
    url: str


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analizar")
async def analizar_archivo(file: UploadFile = File(...)):
    """
    Recibe una imagen como archivo multipart/form-data.
    Devuelve análisis completo en JSON.
    """
    contenido = await file.read()

    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande (máx 15 MB).")

    # Validar MIME por magic bytes (no por nombre)
    mime = _detectar_mime(contenido)
    if mime not in MIMES_PERMITIDOS:
        raise HTTPException(415, f"Formato no soportado: {mime}. Usa JPG, PNG, WebP o GIF.")

    try:
        resultado = analizar_imagen_completa(contenido)
    except Exception as e:
        raise HTTPException(500, f"Error al analizar imagen: {str(e)}")

    return resultado


@app.post("/analizar-url")
def analizar_desde_url(req: UrlRequest):
    """
    Recibe una URL de imagen en JSON: {"url": "https://..."}
    Descarga la imagen y devuelve análisis completo en JSON.
    """
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL inválida. Debe empezar con http:// o https://")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ImageAnalyzerBot/1.0"},
        )
        with urllib.request.urlopen(request, timeout=12) as resp:
            contenido = resp.read(MAX_BYTES + 1)
    except urllib.error.URLError as e:
        raise HTTPException(400, f"No se pudo descargar la imagen: {str(e)}")

    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande (máx 15 MB).")

    mime = _detectar_mime(contenido)
    if mime not in MIMES_PERMITIDOS:
        raise HTTPException(415, f"Formato no soportado: {mime}.")

    try:
        resultado = analizar_imagen_completa(contenido)
    except Exception as e:
        raise HTTPException(500, f"Error al analizar imagen: {str(e)}")

    return resultado


# ──────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────

def _detectar_mime(data: bytes) -> str:
    """Detecta MIME por magic bytes (primeros bytes del archivo)."""
    if data[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:4] in (b'RIFF', b'RIFF') and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    # fallback
    return "application/octet-stream"

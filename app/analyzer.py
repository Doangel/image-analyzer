"""
analyzer.py
Extracción profunda de características visuales de una imagen.
Detecta: colores, tonos de piel, cabello, ropa, figura/silueta,
composición, iluminación, texturas, cantidad de personajes y más.
"""

import io
import math
import numpy as np
from PIL import Image, ImageStat
from colorthief import ColorThief
from scipy import ndimage
from skimage import feature, measure, morphology, filters
import cv2


# ──────────────────────────────────────────────
# TABLAS DE REFERENCIA
# ──────────────────────────────────────────────

COLOR_NAMES_ES = [
    # (R_min, R_max, G_min, G_max, B_min, B_max, nombre)
    (200, 255, 200, 255, 200, 255, "blanco"),
    (0,   60,  0,   60,  0,   60,  "negro"),
    (80,  160, 80,  160, 80,  160, "gris"),
    (160, 220, 160, 220, 160, 220, "gris claro"),
    (40,  80,  40,  80,  40,  80,  "gris oscuro"),
    (180, 255, 0,   80,  0,   80,  "rojo"),
    (200, 255, 100, 180, 0,   80,  "naranja"),
    (180, 255, 150, 220, 0,   80,  "amarillo"),
    (0,   80,  150, 255, 0,   80,  "verde"),
    (0,   80,  80,  180, 0,   80,  "verde oscuro"),
    (0,   100, 180, 255, 180, 255, "cyan"),
    (0,   80,  0,   80,  150, 255, "azul"),
    (0,   60,  0,   60,  80,  160, "azul oscuro"),
    (100, 200, 0,   80,  180, 255, "violeta"),
    (180, 255, 0,   80,  150, 255, "magenta"),
    (200, 255, 160, 220, 140, 200, "beige"),
    (180, 230, 120, 170, 80,  130, "marrón claro"),
    (100, 160, 60,  110, 30,  80,  "marrón"),
    (50,  100, 25,  65,  10,  50,  "marrón oscuro"),
    (255, 255, 200, 230, 150, 190, "amarillo pálido"),
    (200, 240, 200, 240, 220, 255, "lavanda claro"),
    (130, 180, 100, 160, 180, 240, "lila"),
    (160, 210, 100, 150, 180, 240, "violeta claro"),
    (200, 240, 170, 210, 210, 255, "lila pálido"),
    (220, 255, 180, 220, 180, 220, "blanco rosado"),
    (240, 255, 220, 245, 200, 230, "blanco cálido"),
    (200, 240, 220, 255, 220, 255, "blanco frío"),
    (180, 220, 140, 180, 140, 180, "gris violáceo"),
    (160, 200, 160, 200, 180, 220, "gris azulado"),
    (220, 255, 200, 240, 180, 220, "rosa pálido"),
    (220, 255, 100, 160, 100, 160, "rosa"),
    (200, 250, 50,  120, 50,  120, "rosa fuerte"),
    (255, 255, 255, 255, 255, 255, "blanco puro"),
]

HAIR_COLOR_NAMES = {
    "blanco": ["blanco", "blanco puro", "blanco cálido", "blanco frío", "blanco rosado"],
    "plateado": ["gris claro", "gris", "blanco", "lila pálido", "gris violáceo", "gris azulado"],
    "negro": ["negro", "negro azulado"],
    "castaño": ["marrón", "marrón claro", "marrón oscuro"],
    "rubio": ["amarillo", "beige", "amarillo pálido"],
    "pelirrojo": ["naranja", "rojo", "naranja rojizo"],
    "azul": ["azul", "azul oscuro", "cyan"],
    "morado": ["violeta", "violeta claro", "lila", "magenta"],
    "rosa": ["rosa", "rosa pálido", "rosa fuerte"],
    "verde": ["verde", "verde oscuro"],
}

SKIN_TONE_NAMES = {
    "muy claro": (210, 255, 180, 230, 160, 210),
    "claro": (190, 240, 150, 210, 120, 180),
    "medio": (160, 210, 110, 170, 80, 140),
    "oliváceo": (130, 190, 100, 160, 60, 120),
    "oscuro": (80, 150, 50, 110, 30, 90),
    "muy oscuro": (30, 90, 20, 70, 10, 55),
}


# ──────────────────────────────────────────────
# HELPERS DE COLOR
# ──────────────────────────────────────────────

def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    h = 0
    if df != 0:
        if mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        else:
            h = (60 * ((r - g) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx
    v = mx
    return h, s, v


def nombre_color_rgb(r, g, b):
    mejor = "desconocido"
    mejor_dist = float("inf")
    for entry in COLOR_NAMES_ES:
        rmin, rmax, gmin, gmax, bmin, bmax, nombre = entry
        if rmin <= r <= rmax and gmin <= g <= gmax and bmin <= b <= bmax:
            cr = (rmin + rmax) / 2
            cg = (gmin + gmax) / 2
            cb = (bmin + bmax) / 2
            dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = nombre
    if mejor == "desconocido":
        # fallback por componente dominante
        h, s, v = rgb_to_hsv(r, g, b)
        if v < 0.2:
            return "negro"
        if v > 0.85 and s < 0.15:
            return "blanco"
        if s < 0.15:
            return "gris"
        if h < 30 or h > 330:
            return "rojo"
        if h < 60:
            return "naranja"
        if h < 90:
            return "amarillo"
        if h < 150:
            return "verde"
        if h < 210:
            return "cyan"
        if h < 270:
            return "azul"
        if h < 300:
            return "violeta"
        return "magenta"
    return mejor


def es_tono_piel(r, g, b):
    """Detecta si un pixel es tono de piel humano (incluye anime)."""
    h, s, v = rgb_to_hsv(r, g, b)
    # Rango ampliado para incluir estilos de anime (piel muy pálida)
    if v < 0.3:
        return False
    if s < 0.05 and v > 0.85:
        return False  # blanco puro, no piel
    # Tonos cálidos con s moderada
    if 0 <= h <= 40 and 0.08 <= s <= 0.75 and v >= 0.35:
        return True
    if 340 <= h <= 360 and 0.08 <= s <= 0.55 and v >= 0.35:
        return True
    # Piel muy pálida estilo anime
    if 15 <= h <= 50 and 0.03 <= s <= 0.35 and v >= 0.80:
        return True
    return False


def clasificar_cabello(colores_dominantes):
    """Intenta clasificar el color de cabello a partir de la paleta."""
    resultados = []
    for c in colores_dominantes:
        nombre = c["nombre"]
        for categoria, sinónimos in HAIR_COLOR_NAMES.items():
            for s in sinónimos:
                if s in nombre or nombre in s:
                    resultados.append(categoria)
                    break
    if not resultados:
        return "indefinido"
    # el más frecuente
    from collections import Counter
    return Counter(resultados).most_common(1)[0][0]


def describir_tono_piel(r, g, b):
    mejor = "claro"
    mejor_dist = float("inf")
    for nombre, (rmin, rmax, gmin, gmax, bmin, bmax) in SKIN_TONE_NAMES.items():
        if rmin <= r <= rmax and gmin <= g <= gmax and bmin <= b <= bmax:
            cr = (rmin + rmax) / 2
            cg = (gmin + gmax) / 2
            cb = (bmin + bmax) / 2
            dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = nombre
    return mejor


# ──────────────────────────────────────────────
# ANÁLISIS DE FIGURA / SILUETA
# ──────────────────────────────────────────────

def analizar_figura(img_cv):
    """
    Detecta siluetas y proporciones de figuras en la imagen.
    Devuelve: cantidad estimada de figuras, proporción cuerpo, posición.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detección de bordes con Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos significativos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    figuras = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.02:  # ignorar ruido pequeño
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = ch / (cw + 1e-6)  # alto/ancho

        # Clasificar por proporción
        if aspect > 1.8:
            tipo = "figura vertical (personaje de pie)"
        elif aspect > 0.9:
            tipo = "figura cuadrada (personaje sentado o medio cuerpo)"
        else:
            tipo = "figura horizontal (paisaje o elemento de fondo)"

        # Posición relativa en la imagen
        cx_rel = (x + cw / 2) / w
        cy_rel = (y + ch / 2) / h

        if cx_rel < 0.33:
            pos_h = "izquierda"
        elif cx_rel > 0.66:
            pos_h = "derecha"
        else:
            pos_h = "centro"

        if cy_rel < 0.4:
            pos_v = "arriba"
        elif cy_rel > 0.7:
            pos_v = "abajo"
        else:
            pos_v = "medio"

        figuras.append({
            "tipo": tipo,
            "posicion": f"{pos_v} {pos_h}",
            "proporcion_alto_ancho": round(aspect, 2),
            "porcentaje_imagen": round(area / img_area * 100, 1),
        })

    # Estimar cantidad de personajes por análisis de distribución vertical
    # (columnas de bordes densos = personajes separados)
    col_density = np.sum(edges, axis=0)
    threshold = col_density.max() * 0.3
    dense_cols = col_density > threshold
    # Contar grupos continuos de columnas densas
    labeled, num_groups = ndimage.label(dense_cols)
    grupos_grandes = 0
    for i in range(1, num_groups + 1):
        size = np.sum(labeled == i)
        if size > w * 0.06:  # grupo ocupa >6% del ancho
            grupos_grandes += 1

    cantidad_estimada = max(1, min(grupos_grandes, 6))

    return {
        "cantidad_personajes_estimada": cantidad_estimada,
        "figuras_detectadas": len(figuras),
        "detalle_figuras": figuras[:5],  # máx 5 para no saturar
    }


# ──────────────────────────────────────────────
# ANÁLISIS DE COMPOSICIÓN
# ──────────────────────────────────────────────

def analizar_composicion(img_pil):
    """Regla de tercios, zonas de interés, encuadre."""
    w, h = img_pil.size
    arr = np.array(img_pil.convert("L"))

    # Dividir en 9 zonas (3x3)
    zonas = {}
    nombres_zona = [
        ["superior izquierda", "superior centro", "superior derecha"],
        ["medio izquierda",    "centro",           "medio derecha"],
        ["inferior izquierda", "inferior centro",  "inferior derecha"],
    ]
    brillo_zonas = []
    for fy in range(3):
        for fx in range(3):
            y0, y1 = int(h * fy / 3), int(h * (fy + 1) / 3)
            x0, x1 = int(w * fx / 3), int(w * (fx + 1) / 3)
            zona = arr[y0:y1, x0:x1]
            brillo_zonas.append(float(np.mean(zona)))

    zona_mas_brillante = nombres_zona[
        brillo_zonas.index(max(brillo_zonas)) // 3][
        brillo_zonas.index(max(brillo_zonas)) % 3]

    # Tipo de plano por proporción
    if w / h > 1.6:
        plano = "panorámico"
    elif w / h > 1.1:
        plano = "horizontal"
    elif h / w > 1.6:
        plano = "vertical (retrato)"
    else:
        plano = "cuadrado"

    # Encuadre estimado
    # Si la zona central tiene mucho brillo/detalle: primer plano / close-up
    centro_brillo = brillo_zonas[4]
    borde_brillo = np.mean([brillo_zonas[i] for i in [0, 1, 2, 6, 7, 8]])

    if abs(centro_brillo - borde_brillo) > 40:
        encuadre = "primer plano (sujeto centrado)"
    elif borde_brillo > centro_brillo:
        encuadre = "plano abierto (fondo prominente)"
    else:
        encuadre = "plano medio"

    return {
        "tipo_plano": plano,
        "encuadre_estimado": encuadre,
        "zona_mas_iluminada": zona_mas_brillante,
        "resolucion": f"{w}x{h}",
    }


# ──────────────────────────────────────────────
# ANÁLISIS DE ILUMINACIÓN Y AMBIENTE
# ──────────────────────────────────────────────

def analizar_iluminacion(img_pil):
    arr = np.array(img_pil.convert("RGB")).astype(float)
    brillo_global = float(np.mean(arr))
    contraste = float(np.std(arr))

    # Temperatura de color (comparar canales R vs B)
    r_mean = float(np.mean(arr[:, :, 0]))
    g_mean = float(np.mean(arr[:, :, 1]))
    b_mean = float(np.mean(arr[:, :, 2]))

    if r_mean > b_mean * 1.15:
        temp_color = "cálida (tonos naranja/amarillo)"
    elif b_mean > r_mean * 1.15:
        temp_color = "fría (tonos azul/violeta)"
    else:
        temp_color = "neutra"

    if brillo_global < 60:
        ambiente = "muy oscuro / nocturno"
    elif brillo_global < 110:
        ambiente = "oscuro / interior tenue"
    elif brillo_global < 165:
        ambiente = "iluminación moderada"
    elif brillo_global < 210:
        ambiente = "luminoso / exterior día"
    else:
        ambiente = "muy brillante / sobreexpuesto"

    return {
        "brillo_global": round(brillo_global, 1),
        "contraste": round(contraste, 1),
        "temperatura_color": temp_color,
        "ambiente_luminoso": ambiente,
        "canales_rgb_promedio": {
            "rojo": round(r_mean, 1),
            "verde": round(g_mean, 1),
            "azul": round(b_mean, 1),
        },
    }


# ──────────────────────────────────────────────
# ANÁLISIS DE TEXTURA Y DETALLE
# ──────────────────────────────────────────────

def analizar_textura(img_pil):
    arr_gray = np.array(img_pil.convert("L"))
    # Bordes Canny (complejidad de detalle)
    edges = feature.canny(arr_gray, sigma=1.5)
    densidad_bordes = float(np.sum(edges) / edges.size)

    # Gradiente Sobel (sharpness)
    sobelx = filters.sobel_h(arr_gray.astype(float))
    sobely = filters.sobel_v(arr_gray.astype(float))
    sharpness = float(np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)))

    if densidad_bordes > 0.12:
        nivel_detalle = "muy detallado"
    elif densidad_bordes > 0.06:
        nivel_detalle = "detalle medio-alto"
    elif densidad_bordes > 0.03:
        nivel_detalle = "detalle moderado"
    else:
        nivel_detalle = "simple / minimalista"

    if sharpness > 25:
        nitidez = "muy nítido"
    elif sharpness > 12:
        nitidez = "nítido"
    elif sharpness > 5:
        nitidez = "moderado"
    else:
        nitidez = "difuso / desenfocado"

    return {
        "nivel_detalle": nivel_detalle,
        "nitidez": nitidez,
        "densidad_bordes": round(densidad_bordes, 4),
        "sharpness_score": round(sharpness, 2),
    }


# ──────────────────────────────────────────────
# ANÁLISIS DE COLORES DOMINANTES
# ──────────────────────────────────────────────

def analizar_colores(img_bytes):
    ct = ColorThief(io.BytesIO(img_bytes))
    paleta = ct.get_palette(color_count=8, quality=1)

    colores = []
    for r, g, b in paleta:
        h, s, v = rgb_to_hsv(r, g, b)
        colores.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "nombre": nombre_color_rgb(r, g, b),
            "saturacion": round(s, 3),
            "brillo": round(v, 3),
        })

    # Detectar si hay tonos de piel
    tonos_piel = []
    for c in colores:
        r, g, b = c["rgb"]
        if es_tono_piel(r, g, b):
            tonos_piel.append({
                "hex": c["hex"],
                "tono": describir_tono_piel(r, g, b),
            })

    return {
        "paleta_dominante": colores,
        "cantidad_colores_analizados": len(colores),
        "tonos_piel_detectados": tonos_piel,
        "tiene_piel_visible": len(tonos_piel) > 0,
        "color_cabello_inferido": clasificar_cabello(colores),
    }


# ──────────────────────────────────────────────
# GENERADOR DE TAGS Y DESCRIPCIÓN
# ──────────────────────────────────────────────

def generar_tags_y_descripcion(colores_data, figura_data, composicion_data,
                                iluminacion_data, textura_data):
    tags = set()

    # Tags de colores
    for c in colores_data["paleta_dominante"][:5]:
        tags.add(c["nombre"])

    # Tags de piel
    for tp in colores_data["tonos_piel_detectados"]:
        tags.add(f"piel {tp['tono']}")
        tags.add("personaje")

    # Tag de cabello
    cabello = colores_data["color_cabello_inferido"]
    if cabello != "indefinido":
        tags.add(f"cabello {cabello}")
        tags.add(f"pelo {cabello}")

    # Tags de figura
    n = figura_data["cantidad_personajes_estimada"]
    if n == 1:
        tags.update(["un personaje", "personaje solitario"])
    elif n == 2:
        tags.update(["dos personajes", "pareja de personajes"])
    elif n >= 3:
        tags.update([f"{n} personajes", "grupo de personajes", "varios personajes"])

    for fig in figura_data["detalle_figuras"]:
        if "vertical" in fig["tipo"]:
            tags.update(["personaje de pie", "figura completa"])
        elif "cuadrada" in fig["tipo"]:
            tags.update(["medio cuerpo", "retrato"])

    # Tags de composición
    tags.add(composicion_data["tipo_plano"])
    tags.add(composicion_data["encuadre_estimado"])

    # Tags de iluminación
    tags.add(iluminacion_data["ambiente_luminoso"])
    tags.add(iluminacion_data["temperatura_color"])

    # Tags de textura
    tags.add(textura_data["nivel_detalle"])
    tags.add(textura_data["nitidez"])

    # Descripción narrativa
    partes = []

    n_str = {1: "un personaje", 2: "dos personajes", 3: "tres personajes"}.get(n, f"{n} personajes")
    partes.append(f"Imagen con {n_str}.")

    if colores_data["tiene_piel_visible"]:
        tono = colores_data["tonos_piel_detectados"][0]["tono"] if colores_data["tonos_piel_detectados"] else "claro"
        partes.append(f"Tono de piel {tono}.")

    cabello_desc = colores_data["color_cabello_inferido"]
    if cabello_desc != "indefinido":
        partes.append(f"Cabello {cabello_desc}.")

    colores_principales = [c["nombre"] for c in colores_data["paleta_dominante"][:3]]
    partes.append(f"Colores predominantes: {', '.join(colores_principales)}.")

    enc = composicion_data["encuadre_estimado"]
    partes.append(f"Encuadre: {enc}.")

    partes.append(f"Iluminación {iluminacion_data['ambiente_luminoso']}, temperatura {iluminacion_data['temperatura_color']}.")
    partes.append(f"Imagen {textura_data['nivel_detalle']} y {textura_data['nitidez']}.")

    descripcion = " ".join(partes)

    return {
        "tags": ", ".join(sorted(tags)),
        "tags_lista": sorted(tags),
        "descripcion": descripcion,
    }


# ──────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ──────────────────────────────────────────────

def analizar_imagen_completa(img_bytes: bytes) -> dict:
    """
    Punto de entrada principal.
    Recibe bytes de imagen, devuelve dict con todos los análisis.
    """
    # Cargar con Pillow
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Limitar tamaño para velocidad (mantener aspect ratio)
    max_dim = 800
    w, h = img_pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convertir a OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Re-encodear a bytes para ColorThief (necesita bytes frescos)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    img_bytes_fresh = buf.getvalue()

    # Ejecutar todos los análisis
    colores_data     = analizar_colores(img_bytes_fresh)
    figura_data      = analizar_figura(img_cv)
    composicion_data = analizar_composicion(img_pil)
    iluminacion_data = analizar_iluminacion(img_pil)
    textura_data     = analizar_textura(img_pil)
    tags_data        = generar_tags_y_descripcion(
        colores_data, figura_data, composicion_data,
        iluminacion_data, textura_data
    )

    return {
        "colores":     colores_data,
        "figura":      figura_data,
        "composicion": composicion_data,
        "iluminacion": iluminacion_data,
        "textura":     textura_data,
        "tags":        tags_data["tags"],
        "tags_lista":  tags_data["tags_lista"],
        "descripcion": tags_data["descripcion"],
        "resolucion_original": f"{w}x{h}",
    }

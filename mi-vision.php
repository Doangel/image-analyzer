<?php
/**
 * mi-vision.php
 * Reemplaza gemini-vision.php.
 * Llama a tu propia API de análisis de imágenes en Render.
 *
 * USO:
 *   require_once 'mi-vision.php';
 *   $resultado = analizarImagenConGemini($ruta, '', 'archivo');
 *   // o
 *   $resultado = analizarImagenConGemini($url, '', 'url');
 *
 * Devuelve: ['tags' => string, 'ai_description' => string|null]
 */

// ────────────────────────────────────────────────────────────────
// CONFIGURACIÓN — cambia esta URL por la de tu app en Render
// ────────────────────────────────────────────────────────────────
define('VISION_API_BASE', 'https://TU-APP.onrender.com');
//                                  ↑ reemplaza esto con tu URL real


/**
 * Analiza una imagen local o una URL externa.
 *
 * @param string $rutaOUrl  Ruta local (/home/.../pinesimg/pin_xxx.webp) o URL https://...
 * @param string $_         Parámetro ignorado (compatibilidad con firma original)
 * @param string $modo      'archivo' | 'url'
 * @return array            ['tags' => string, 'ai_description' => string|null]
 */
function analizarImagenConGemini(string $rutaOUrl, string $_, string $modo): array
{
    if ($modo === 'url') {
        return _llamarApiUrl($rutaOUrl);
    }
    return _llamarApiArchivo($rutaOUrl);
}


// ────────────────────────────────────────────────────────────────
// FUNCIONES INTERNAS
// ────────────────────────────────────────────────────────────────

/**
 * Envía un archivo local a /analizar (multipart/form-data).
 */
function _llamarApiArchivo(string $ruta): array
{
    if (!file_exists($ruta) || !is_readable($ruta)) {
        error_log("[mi-vision] Archivo no encontrado o no legible: $ruta");
        return ['tags' => '', 'ai_description' => null];
    }

    $mime = _detectarMime($ruta);
    $cfile = new CURLFile($ruta, $mime, basename($ruta));

    $ch = curl_init(VISION_API_BASE . '/analizar');
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => ['file' => $cfile],
        CURLOPT_TIMEOUT        => 60,   // análisis puede tardar en servidor frío
        CURLOPT_SSL_VERIFYPEER => true,
    ]);

    $resp   = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err    = curl_error($ch);
    curl_close($ch);

    if ($err || $status !== 200) {
        error_log("[mi-vision] Error API archivo: HTTP $status — $err");
        return ['tags' => '', 'ai_description' => null];
    }

    return _parsearRespuesta($resp);
}


/**
 * Envía una URL externa a /analizar-url (JSON body).
 */
function _llamarApiUrl(string $url): array
{
    $payload = json_encode(['url' => $url]);

    $ch = curl_init(VISION_API_BASE . '/analizar-url');
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => $payload,
        CURLOPT_HTTPHEADER     => ['Content-Type: application/json'],
        CURLOPT_TIMEOUT        => 60,
        CURLOPT_SSL_VERIFYPEER => true,
    ]);

    $resp   = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $err    = curl_error($ch);
    curl_close($ch);

    if ($err || $status !== 200) {
        error_log("[mi-vision] Error API url: HTTP $status — $err");
        return ['tags' => '', 'ai_description' => null];
    }

    return _parsearRespuesta($resp);
}


/**
 * Convierte la respuesta JSON de la API al formato esperado por subir-img.php.
 */
function _parsearRespuesta(string $json): array
{
    $data = json_decode($json, true);
    if (!$data) {
        return ['tags' => '', 'ai_description' => null];
    }

    // Datos principales
    $tags        = $data['tags']        ?? '';
    $descripcion = $data['descripcion'] ?? null;

    // Enriquecer tags con datos adicionales de la respuesta
    $extras = [];

    // Cantidad de personajes
    $n = $data['figura']['cantidad_personajes_estimada'] ?? 0;
    if ($n > 0) {
        $extras[] = "$n personaje" . ($n !== 1 ? 's' : '');
    }

    // Color de cabello
    $cabello = $data['colores']['color_cabello_inferido'] ?? '';
    if ($cabello && $cabello !== 'indefinido') {
        $extras[] = "cabello $cabello";
        $extras[] = "pelo $cabello";
    }

    // Piel visible
    if (!empty($data['colores']['tiene_piel_visible'])) {
        $extras[] = 'piel visible';
        foreach (($data['colores']['tonos_piel_detectados'] ?? []) as $tp) {
            $extras[] = 'piel ' . ($tp['tono'] ?? '');
        }
    }

    // Encuadre
    $enc = $data['composicion']['encuadre_estimado'] ?? '';
    if ($enc) $extras[] = $enc;

    // Ambiente
    $amb = $data['iluminacion']['ambiente_luminoso'] ?? '';
    if ($amb) $extras[] = $amb;

    // Combinar tags
    $todos = array_filter(array_unique(array_merge(
        array_map('trim', explode(',', $tags)),
        $extras
    )));

    return [
        'tags'           => implode(', ', $todos),
        'ai_description' => $descripcion,
        // Datos extendidos opcionales (puedes guardarlos en columnas extra si quieres)
        '_colores'        => $data['colores']     ?? null,
        '_figura'         => $data['figura']      ?? null,
        '_composicion'    => $data['composicion'] ?? null,
        '_iluminacion'    => $data['iluminacion'] ?? null,
        '_textura'        => $data['textura']     ?? null,
    ];
}


/**
 * Detecta el MIME type de un archivo local por magic bytes.
 */
function _detectarMime(string $ruta): string
{
    $handle = fopen($ruta, 'rb');
    $header = fread($handle, 12);
    fclose($handle);

    if (substr($header, 0, 3) === "\xff\xd8\xff")      return 'image/jpeg';
    if (substr($header, 0, 8) === "\x89PNG\r\n\x1a\n") return 'image/png';
    if (substr($header, 8, 4) === 'WEBP')               return 'image/webp';
    if (substr($header, 0, 6) === 'GIF87a' ||
        substr($header, 0, 6) === 'GIF89a')             return 'image/gif';

    // Fallback con finfo si está disponible
    if (function_exists('finfo_open')) {
        $fi   = finfo_open(FILEINFO_MIME_TYPE);
        $mime = finfo_file($fi, $ruta);
        finfo_close($fi);
        return $mime ?: 'image/jpeg';
    }

    return 'image/jpeg';
}

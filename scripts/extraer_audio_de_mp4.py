import os
import subprocess

def extraer_pistas_audio(mp4_path, extract_only = None):
    """
    Extrae todas las pistas de audio de un archivo MP4 usando ffmpeg.
    Guarda cada pista en /data/raw/[nombre_del_video]_[numero_pista].[extension]
    extract_only: Si es None, extrae todas las pistas. Si es un entero, extrae solo esa pista (0-indexed).
    """
    # Obtener nombre base del archivo sin extensión
    nombre_video = os.path.splitext(os.path.basename(mp4_path))[0]
    # Crear carpeta de salida si no existe
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'audio', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extrayendo pistas de audio de {mp4_path}...")
    # Obtener información de las pistas de audio usando ffprobe
    cmd_info = [
        'ffprobe', '-v', 'error', '-select_streams', 'a',
        '-show_entries', 'stream=index,codec_name',
        '-of', 'csv=p=0', mp4_path
    ]
    result = subprocess.run(cmd_info, capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if not lines or lines == ['']:
        print("No se encontraron pistas de audio.")
        return
    for idx, line in enumerate(lines):
        if extract_only is not None and idx != extract_only:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        stream_index, codec = parts
        ext = 'm4a' if codec == 'aac' else codec
        output_file = os.path.join(
            output_dir, f"{nombre_video}_{idx}.{ext}"
        )
        cmd_extract = [
            'ffmpeg', '-y', '-i', mp4_path,
            '-map', f'0:a:{idx}', '-vn', '-acodec', 'copy', '-ac', '1', output_file
        ]
        subprocess.run(cmd_extract, check=True)
        print(f"Pista {idx} extraída: {output_file}")


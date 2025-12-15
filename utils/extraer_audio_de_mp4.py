import os
import subprocess

def extraer_pistas_audio(mp4_path, extract_only=None, output_format='m4a'):
    """
    Extrae todas las pistas de audio de un archivo MP4 usando ffmpeg.
    Guarda cada pista en /audio/raw/[nombre_del_video]_[numero_pista].[extension]
    extract_only: Si es None, extrae todas las pistas. Si es un entero, extrae solo esa pista (0-indexed).
    output_format: 'm4a', 'wav' o 'mp3'
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

        # Elegir extensión y parámetros según formato de salida
        if output_format == 'wav':
            ext = 'wav'
            acodec = 'pcm_s16le'
            extra_args = ['-ar', '44100']  # 44.1kHz
        elif output_format == 'mp3':
            ext = 'mp3'
            acodec = 'libmp3lame'
            extra_args = ['-ab', '192k']
        else:  # m4a (default)
            ext = 'm4a' if codec == 'aac' else codec
            acodec = 'copy'
            extra_args = []

        output_file = os.path.join(
            output_dir, f"{nombre_video}_{idx}.{ext}"
        )
        cmd_extract = [
            'ffmpeg', '-y', '-i', mp4_path,
            '-map', f'0:a:{idx}', '-vn', '-ac', '1'
        ]
        if acodec == 'copy':
            cmd_extract += ['-acodec', 'copy']
        else:
            cmd_extract += ['-acodec', acodec]
        cmd_extract += extra_args
        cmd_extract.append(output_file)

        subprocess.run(cmd_extract, check=True)
        print(f"Pista {idx} extraída: {output_file}")

def extraer_pistas_audio_full_dir(extract_only=None, output_format='m4a'):
    """
    Recorre todos los archivos .mp4 en /videos y extrae sus pistas de audio.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    videos_dir = os.path.join(base_dir, 'videos')
    if not os.path.isdir(videos_dir):
        print(f"No existe el directorio de videos: {videos_dir}")
        return

    mp4s = [f for f in os.listdir(videos_dir) if f.lower().endswith('.mp4')]
    if not mp4s:
        print("No hay archivos .mp4 en /videos.")
        return

    for fname in mp4s:
        mp4_path = os.path.join(videos_dir, fname)
        try:
            extraer_pistas_audio(mp4_path, extract_only=extract_only, output_format=output_format)
        except Exception as e:
            print(f"Error procesando {mp4_path}: {e}")


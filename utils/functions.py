import os
import shutil


def copy_latest_run_models(source_root="output/model", target_dir="models/generated/glowtts"):
    """
    Busca la última ejecución de entrenamiento en 'output/model/' que contenga
    'best_model.pth' y 'config.json'. Si el 'best_model.pth' del source es más
    nuevo que el del target (o el target no existe), copia ambos al target_dir.
    Retorna las rutas destino (best_model, config) o (None, None) si no hay nada que copiar.
    """
    try:
        run_candidates = []
        for name in os.listdir(source_root):
            run_path = os.path.join(source_root, name)
            if not os.path.isdir(run_path):
                continue
            best_model = os.path.join(run_path, "best_model.pth")
            config_json = os.path.join(run_path, "config.json")
            if os.path.isfile(best_model) and os.path.isfile(config_json):
                mtime = max(
                    os.path.getmtime(best_model),
                    os.path.getmtime(config_json),
                    os.path.getmtime(run_path),
                )
                run_candidates.append((mtime, run_path, best_model, config_json))

        if not run_candidates:
            print("No se encontró ninguna ejecución con best_model.pth y config.json en 'output/model/'.")
            return None, None

        # Seleccionar el source más reciente
        run_candidates.sort(key=lambda x: x[0], reverse=True)
        _, latest_run, latest_best_model, latest_config = run_candidates[0]

        os.makedirs(target_dir, exist_ok=True)
        dst_best = os.path.join(target_dir, "best_model.pth")
        dst_cfg = os.path.join(target_dir, "config.json")

        # Determinar si hay que copiar por ausencia o por ser más nuevo
        src_best_mtime = os.path.getmtime(latest_best_model)
        dst_best_exists = os.path.isfile(dst_best)
        dst_cfg_exists = os.path.isfile(dst_cfg)
        dst_best_mtime = os.path.getmtime(dst_best) if dst_best_exists else -1

        if (not dst_best_exists or not dst_cfg_exists) or (src_best_mtime > dst_best_mtime):
            shutil.copy2(latest_best_model, dst_best)
            shutil.copy2(latest_config, dst_cfg)
            print(f"Copiados:\n- {latest_best_model} -> {dst_best}\n- {latest_config} -> {dst_cfg}")
            return dst_best, dst_cfg
        else:
            print("Target ya está actualizado. No se realizaron copias.")
            return dst_best, dst_cfg
    except Exception as e:
        print(f"Error al copiar modelos: {e}")
        return None, None

def copy_latest_vocoder_models(source_root="output/vocoder", target_dir="models/generated/vocoder"):
    """
    Busca la última ejecución de entrenamiento en 'output/vocoder/' que contenga
    'best_model.pth' y 'config.json'. Si el 'best_model.pth' del source es más
    nuevo que el del target (o el target no existe), copia ambos al target_dir.
    Retorna las rutas destino (best_model, config) o (None, None) si no hay nada que copiar.
    """
    try:
        run_candidates = []
        for name in os.listdir(source_root):
            run_path = os.path.join(source_root, name)
            if not os.path.isdir(run_path):
                continue
            best_model = os.path.join(run_path, "best_model.pth")
            config_json = os.path.join(run_path, "config.json")
            if os.path.isfile(best_model) and os.path.isfile(config_json):
                mtime = max(
                    os.path.getmtime(best_model),
                    os.path.getmtime(config_json),
                    os.path.getmtime(run_path),
                )
                run_candidates.append((mtime, run_path, best_model, config_json))

        if not run_candidates:
            print("No se encontró ninguna ejecución con best_model.pth y config.json en 'output/vocoder/'.")
            return None, None

        run_candidates.sort(key=lambda x: x[0], reverse=True)
        _, latest_run, latest_best_model, latest_config = run_candidates[0]

        os.makedirs(target_dir, exist_ok=True)
        dst_best = os.path.join(target_dir, "best_model.pth")
        dst_cfg = os.path.join(target_dir, "config.json")

        src_best_mtime = os.path.getmtime(latest_best_model)
        dst_best_exists = os.path.isfile(dst_best)
        dst_cfg_exists = os.path.isfile(dst_cfg)
        dst_best_mtime = os.path.getmtime(dst_best) if dst_best_exists else -1

        if (not dst_best_exists or not dst_cfg_exists) or (src_best_mtime > dst_best_mtime):
            shutil.copy2(latest_best_model, dst_best)
            shutil.copy2(latest_config, dst_cfg)
            print(f"Copiados (vocoder):\n- {latest_best_model} -> {dst_best}\n- {latest_config} -> {dst_cfg}")
            return dst_best, dst_cfg
        else:
            print("Vocoder target ya está actualizado. No se realizaron copias.")
            return dst_best, dst_cfg
    except Exception as e:
        print(f"Error al copiar modelos del vocoder: {e}")
        return None, None

from PIL import Image

im = Image.open("e:\\Documentos\\gifs_drops\\rustmas\\gifs\\con fondo y efecto\\final.gif")
duration = 0
for frame in range(im.n_frames):
    im.seek(frame)
    duration += im.info['duration']

print("Duración total:", duration / 1000, "segundos")
import matplotlib.pyplot as plt
import numpy as np
import time

# ダミー画像作成
img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

fig, ax = plt.subplots()
im = ax.imshow(img, cmap='gray')
plt.ion()
plt.show()

# 10回画像を更新
for i in range(10):
    img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    im.set_data(img)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.5)

plt.ioff()
plt.show()

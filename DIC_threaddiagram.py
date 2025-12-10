import numpy as np
from collections import deque
from DIC_read_image import BufferManager

def bfs_region_grow(seed_result):
    H, W = BufferManager.refImg.shape
    ROI_list = BufferManager.mask
    seed_valid_result = []
    seed_valid_pos = []
    for (cx, cy, flag, defvector, corrcoef) in seed_result:
        print(f"({cx},{cy}) flag={flag}: {defvector[:2]}, Ncc[{corrcoef}]")
        if flag and corrcoef < 0.1:
            seed_valid_result.append((cx, cy, flag, defvector, corrcoef))
            seed_valid_pos.append((cx, cy))
        else:
            continue
    threaddiagram = -1 * np.ones((H, W), dtype=int)
    queues = [deque() for _ in seed_valid_pos]
    for idx, (x, y) in enumerate(seed_valid_pos):
        threaddiagram[y, x] = idx
        queues[idx].append((y, x))
    # 多源扩散
    while any(queues):
        for i in range(len(seed_valid_pos)):
            if not queues[i]:
                continue
            y, x = queues[i].popleft()
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    for mask in ROI_list:
                        if mask[ny, nx]:
                            break
                    if mask[ny, nx] and threaddiagram[ny, nx] == -1:
                        threaddiagram[ny, nx] = i
                        queues[i].append((ny, nx))
    return threaddiagram, seed_valid_result
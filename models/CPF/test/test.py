import numpy as np

def pc_padding(pc, num_points):
    divisor = pc.shape[0] // num_points
    if divisor > 1:
        pc_pad = pc.repeat(divisor + 1, 0)[:num_points]
    else:
        idx = np.random.choice(pc.shape[0], size=num_points)
        pc_pad = pc[idx]
    return  pc_pad


if __name__ == '__main__':
    pc = np.random.rand(3000, 3)
    pc_padding = pc_padding(pc, 8000)
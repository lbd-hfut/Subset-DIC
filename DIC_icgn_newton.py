import numpy as np
from typing import Tuple
from DIC_read_image import BufferManager

SUCCESS = 1
FAILED = 0

# -------------------------
# B-spline interp
# -------------------------
def interpqbs(xs, ys, REF_FLAG=False, DEF_FLAG=False):
    # 1. 预构建 integer 坐标 (N,2)
    xs_floor = np.floor(xs).astype(int)
    ys_floor = np.floor(ys).astype(int)
    num_pts = len(xs)
    coords_arr = np.stack([ys_floor, xs_floor], axis=1)  # (N,2)
    
    # 2. 构建 QK_B_QKT_arr  (N,6,6)
    QK_B_QKT_arr = np.zeros((num_pts, 6, 6))
    for i in range(num_pts):
        y, x = coords_arr[i]
        if REF_FLAG:
            QK_B_QKT_arr[i] = BufferManager.QKBQKT_ref[(y, x)]
        if DEF_FLAG:
            QK_B_QKT_arr[i] = BufferManager.QKBQKT_def[(y, x)]
    
    # 3. 构造 x_vec、y_vec  (N,6)
    xd = xs - xs_floor  # (N,)
    yd = ys - ys_floor  # (N,)
    x_powers = np.stack([xd**i for i in range(6)], axis=1)  # (N,6)
    y_powers = np.stack([yd**i for i in range(6)], axis=1)  # (N,6)
    
    # 4. 做 y_vec @ M  → shape (N,6)
    tmp = np.einsum("ni,nij->nj", y_powers, QK_B_QKT_arr)
    values = np.einsum("ni,ni->n", tmp, x_powers)
    return values


def construct_w_matrix(defvector):
    # Translation
    u, v = defvector[0], defvector[1]
    # First-order derivatives (affine terms)
    ux, vx, uy, vy = defvector[2], defvector[3], defvector[4], defvector[5]
    # Second-order derivatives
    uxx, vxx, uxy, vxy, uyy, vyy = \
        defvector[6], defvector[7], defvector[8], defvector[9], defvector[10], defvector[11]
    
    # S1–S6 (u-related)
    S1  = 2*ux + ux**2 + u*uxx
    S2  = 2*u*uxy + 2*(1 + ux)*uy
    S3  = uy**2 + u*uyy
    S4  = 2*u*(1 + ux)
    S5  = 2*u*uy
    S6  = u**2

    # S7–S12 (u–v coupling)
    S7  = 0.5*(v*uxx + 2*(1 + ux)*vx + u*vxx)
    S8  = uy*vx + ux*vy + v*uxy + u*vxy + vy + ux
    S9  = 0.5*(v*uyy + 2*uy*(1 + vy) + u*vyy)
    S10 = v + v*ux + u*vx
    S11 = u + v*uy + u*vy
    S12 = u*v

    # S13–S18 (v-related)
    S13 = vx**2 + v*vxx
    S14 = 2*v*vxy + 2*vx*(1 + vy)
    S15 = 2*vy + vy**2 + v*vyy
    S16 = 2*v*vx
    S17 = 2*v*(1 + vy)
    S18 = v**2

    
    
    # ------ A ------
    A =  np.array([
        [1+S1,        S2,       S3],
        [S7,        1+S8,       S9],
        [S13,        S14,       1+S15]
    ])
    # ------ B ------
    B =  np.array([
        [S4,        S5,       S6],
        [S10,       S11,      S12],
        [S16,       S17,      S18]
    ])
    # ------ C ------
    C = np.array([
        [0.5*uxx,  uxy,     0.5*uyy],
        [0.5*vxx,  vxy,     0.5*vyy],
        [0,        0,       0]
    ])
    # ------ D ------
    D = np.array([
        [1+ux,     uy,      u],
        [vx,       1+vy,    v],
        [0,        0,       1]
    ])
    # C_add = np.array([
    #     [0,         2*u,        0],
    #     [0,         v,          u],
    #     [0,         0,          2*v]
    # ])
    # D_add = np.array([
    #     [2*ux,      2*uy,        0],
    #     [vx,        ux+vy,       uy],
    #     [0,         2*vx,        2*vy]
    # ])
    
    # C = C + C_add
    # D = D + D_add
    
    top = np.hstack((A, B))
    bottom = np.hstack((C, D))
    W = np.vstack((top, bottom))
    return W


# -------------------------
# Inverse compositional update 
# -------------------------
def inverse_compositional_update_2nd_order(defvector_old, defvector_delta):
    W_old = construct_w_matrix(defvector_old)
    W_delta = construct_w_matrix(defvector_delta)
    
    # Inverse compositional for first-order
    W_new = W_old @ np.linalg.inv(W_delta)
    
    # Updated first-order parameters
    u, v = W_new[3,5], W_new[4,5]
    ux, vx, uy, vy = W_new[3,3]-1, W_new[4,3], W_new[3,4], W_new[4,4]-1
    uxx, vxx, uxy, vxy, uyy, vyy = \
         W_new[3,0]*2, W_new[4,0]*2, W_new[3,1], W_new[4,1], W_new[3,2]*2, W_new[4,2]*2
    
    defvector_new = np.array([
        u, v, ux, vx, uy, vy, uxx, vxx, uxy, vxy, uyy, vyy
    ])
    
    return defvector_new

# -------------------------
# Newton (IC-GN) implementation
# -------------------------
def iterativesearch(
    defvector_init: np.ndarray,
    xc: np.int32,
    yc: np.int32,
    dx: np.ndarray,
    dy: np.ndarray,
    max_iter: int = 20,
    cutoff_diffnorm: float = 1e-6,
    lambda_reg: float = 1e-3
) -> Tuple[int, np.ndarray]:
    
    X_valid_corrd = xc + dx
    Y_valid_corrd = yc + dy
    
    f_buffer = BufferManager.refImg[Y_valid_corrd, X_valid_corrd]
    fm = np.sum(f_buffer) / len(f_buffer)
    deltaf_inv = np.sqrt(np.sum(((f_buffer - fm)**2)))
    if deltaf_inv < lambda_reg:
        return FAILED, defvector_init, -1
    else:
        deltaf_inv = 1 / deltaf_inv
        df_dp_buffer = np.zeros((len(f_buffer), 12), dtype=np.float32)
        df_dx = BufferManager.fx[Y_valid_corrd, X_valid_corrd]
        df_dy = BufferManager.fy[Y_valid_corrd, X_valid_corrd]
        df_dp_buffer[:, 0] = df_dx
        df_dp_buffer[:, 1] = df_dy
        # for products, use np.multiply with out to avoid creating X*gx temporaries
        np.multiply(dx, df_dx, out=df_dp_buffer[:, 2])   # X * fx
        np.multiply(dx, df_dy, out=df_dp_buffer[:, 3])   # X * fy
        np.multiply(dy, df_dx, out=df_dp_buffer[:, 4])   # Y * fx
        np.multiply(dy, df_dy, out=df_dp_buffer[:, 5])   # Y * fy
        # squared terms
        np.multiply(dx * dx * 0.5, df_dx, out=df_dp_buffer[:, 6])  # 0.5 X^2 * fx
        np.multiply(dx * dx * 0.5, df_dy, out=df_dp_buffer[:, 7])  # 0.5 X^2 * fy
        np.multiply(dx * dy, df_dx, out=df_dp_buffer[:, 8])        # X * Y * fx
        np.multiply(dx * dy, df_dy, out=df_dp_buffer[:, 9])        # X * Y * fy
        np.multiply(dy * dy * 0.5, df_dx, out=df_dp_buffer[:, 10]) # 0.5 Y^2 * fx
        np.multiply(dy * dy * 0.5, df_dy, out=df_dp_buffer[:, 11]) # 0.5 Y^2 * fy
        # compute Hessian
        hessian_gn = df_dp_buffer.T @ df_dp_buffer
        hessian_gn = hessian_gn * 2 * (deltaf_inv**2)
        
        try:
            cholesky_G = np.linalg.cholesky(hessian_gn)
            positivedef = True
        except np.linalg.LinAlgError:
            positivedef = False  # 非正定
            
        if positivedef:
            for iter in range(max_iter):
                flag, defvector_init, diffnorm, corrcoef = newton(
                    defvector_init=defvector_init,
                    xc=xc, yc=yc, dx=dx, dy=dy,
                    df_dp_buffer=df_dp_buffer,
                    f_buffer=f_buffer,
                    fm=fm, deltaf_inv=deltaf_inv,
                    lambda_reg=lambda_reg,
                    cutoff_diffnorm=cutoff_diffnorm,
                    cholesky_G=cholesky_G
                    )
                if diffnorm < cutoff_diffnorm:
                    return SUCCESS, defvector_init, corrcoef
                if flag == FAILED:
                    return FAILED, defvector_init, corrcoef
            return SUCCESS, defvector_init, corrcoef
        else:
            return FAILED, defvector_init, 1.0

# -------------------------
# Newton (IC-GN) implementation
# -------------------------
def newton(
    defvector_init: np.ndarray, 
    xc: np.int32,
    yc: np.int32,
    dx: np.ndarray,
    dy: np.ndarray,
    df_dp_buffer: np.ndarray,
    f_buffer: np.ndarray,
    fm: float,
    deltaf_inv: float,
    lambda_reg: float,
    cutoff_diffnorm: float,
    cholesky_G: np.ndarray
) -> Tuple[int, np.ndarray, float]:
    
    W = construct_w_matrix(defvector_init)
    dX = np.column_stack([dx**2, dx*dy, dy**2, dx, dy, np.ones_like(dx)])
    dX_tilida = np.einsum('ij,nj->ni', W, dX)
    
    X_tilda_corrd = xc + dX_tilida[:, 3]
    Y_tilda_corrd = yc + dX_tilida[:, 4]
    try:
        g_buffer = interpqbs(X_tilda_corrd, Y_tilda_corrd, REF_FLAG=False, DEF_FLAG=True)
    except:
        return FAILED, defvector_init, cutoff_diffnorm*10, 1
    gm = np.sum(g_buffer) / len(g_buffer)
    deltag_inv = np.sqrt(np.sum(((g_buffer - gm)**2)))
    if deltag_inv < lambda_reg:
        return FAILED, defvector_init, cutoff_diffnorm*10, 1
    else:
        deltag_inv = 1 / deltag_inv
        normalized_diff = (f_buffer-fm)*deltaf_inv - (g_buffer-gm)*deltag_inv
        corrcoef = np.sum(normalized_diff**2)
        gradient_buffer = 2 * deltaf_inv * np.einsum('i,ij->j', normalized_diff, df_dp_buffer)
        # gradient_buffer = np.sum(gradient_buffer, axis=0)
        y = np.linalg.solve(cholesky_G, gradient_buffer)       # forward solve
        x = np.linalg.solve(cholesky_G.T, y)     # backward solve
        delta = -x
        # 计算差分范数
        diffnorm = np.linalg.norm(delta) 
        if diffnorm < cutoff_diffnorm:
            return SUCCESS, defvector_init, diffnorm, corrcoef
        else:
            defvector_new = inverse_compositional_update_2nd_order(
                defvector_init, delta)
            # print("defvector_new:")
            # print(defvector_new)
            # print("corrcoef:")
            # print(corrcoef)
            return SUCCESS, defvector_new, diffnorm, corrcoef
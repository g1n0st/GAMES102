import taichi as ti

ti.init(arch = ti.cpu)

max_num_vertex = 1024

num_vertex = ti.field(dtype = ti.i32, shape = ())
p = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex)
t = ti.field(dtype = ti.f32, shape = max_num_vertex) # parameterization

a = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
b = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
c = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
d = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]

'''
 coeffcient of M matrix
 [B0, C0, ...
 [A1, B1, C1, ...
 [0, A2, B2, C2, ...
 [0, ...
 '''
A = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-2]
B = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-2]
C = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-2]
D = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-2]
M = ti.field(dtype = ti.f32, shape = max_num_vertex) # range [0, n-2]

gui = ti.GUI('Spline Test System', res = (512, 512), background_color = 0xDDDDDD)

@ti.func
def TDMA(n):
    C[0] /= B[0]
    D[0] /= B[0]

    i = 1
    while i < n:
        tmp = (B[i] - A[i] * C[i - 1])
        C[i] = C[i] / tmp
        D[i] = (D[i] - A[i] * D[i - 1]) / tmp
        i += 1
    
    M[0], M[n + 1], M[n] = 0, 0, D[n - 1]
    i = n - 2
    while i >= 0:
        M[i + 1] = D[i] - C[i] * M[i + 2]
        i -= 1

@ti.kernel
def eval():
    n = num_vertex[None]
    for i in range(0, n - 1):
        t[i] = p[i + 1][0] - p[i][0]

    for i in range(0, n - 2):
        A[i] = t[i]
        B[i] = 2 * (t[i] + t[i + 1])
        C[i] = t[i + 1]

    for i in range(0, n - 2):
        k1 = (p[i + 2][1] - p[i + 1][1]) / t[i + 1]
        k0 = (p[i + 1][1] - p[i][1]) / t[i]
        D[i] = 6 * (k1 - k0)
    
    TDMA(n - 2)

    for i in range(0, n - 1):
        a[i] = p[i][1]
        b[i] = (p[i + 1][1] - p[i][1]) / t[i] - (2 * t[i] * M[i] + t[i] * M[i + 1]) / 6
        c[i] = M[i] / 2
        d[i] = (M[i + 1] - M[i]) / (6 * t[i])
        print(a[i], b[i], c[i], d[i])

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            num_vertex[None] = 0
        elif e.key == ti.GUI.LMB:
            p[num_vertex[None]] = ti.Vector([e.pos[0], e.pos[1]])
            num_vertex[None] += 1
            if num_vertex[None] >= 3: eval()

    X = p.to_numpy()
    n = num_vertex[None]
    
    k = 50

    # Draw linear interpolation
    for i in range(0, n - 1):
        gui.line(begin = X[i], end = X[i + 1], radius = 2, color = 0x444444)
        delta_t = (X[i + 1][0] - X[i][0]) / k
        for j in range(k):
            x0 = j * delta_t
            x1 = x0 + delta_t
            y0 = a[i] + b[i] * x0 + c[i] * x0 ** 2 + d[i] * x0 ** 3
            y1 = a[i] + b[i] * x1 + c[i] * x1 ** 2 + d[i] * x1 ** 3
            gui.line(begin = (x0 + X[i][0], y0), end = (x1 + X[i][0], y1), radius = 2, color = 0xFF0000)

    # Draw the vertices
    for i in range(n):
        gui.circle(pos = X[i], color = 0x111111, radius = 5)

    gui.show()

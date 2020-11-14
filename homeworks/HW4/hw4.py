import taichi as ti

ti.init(arch = ti.cpu)

max_num_vertex = 1024

num_vertex = ti.field(dtype = ti.i32, shape = ())
p = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex)
t = ti.field(dtype = ti.f32, shape = max_num_vertex) # parameterization

a = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
b = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
c = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]
d = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex) # range [0, n-1]

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

gui = ti.GUI('Spline Test System', res = (1024, 1024), background_color = 0xDDDDDD)

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

@ti.func
def para(p1, p2):
    return ti.sqrt((p2 - p1).norm())

@ti.kernel
def eval(dim : ti.template()):
    n = num_vertex[None]
    for i in range(0, n - 1):
        t[i] = para(p[i], p[i + 1])

    for i in range(0, n - 2):
        A[i] = t[i]
        B[i] = 2 * (t[i] + t[i + 1])
        C[i] = t[i + 1]

    for i in range(0, n - 2):
        k1 = (p[i + 2][dim] - p[i + 1][dim]) / t[i + 1]
        k0 = (p[i + 1][dim] - p[i][dim]) / t[i]
        D[i] = 6 * (k1 - k0)
        
    TDMA(n - 2)

    for i in range(0, n - 1):
        a[i][dim] = p[i][dim]
        b[i][dim] = (p[i + 1][dim] - p[i][dim]) / t[i] - (2 * t[i] * M[i] + t[i] * M[i + 1]) / 6
        c[i][dim] = M[i] / 2
        d[i][dim] = (M[i + 1] - M[i]) / (6 * t[i])

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            num_vertex[None] = 0
        elif e.key == ti.GUI.LMB:
            p[num_vertex[None]] = ti.Vector([e.pos[0], e.pos[1]])
            num_vertex[None] += 1
            if num_vertex[None] >= 2: 
                eval(0)
                eval(1)

    X = p.to_numpy()
    n = num_vertex[None]
    
    k = 50

    # Draw linear interpolation
    for i in range(0, n - 1):
        gui.line(begin = X[i], end = X[i + 1], radius = 2, color = 0x444444)
        delta_t = t[i] / k
        for j in range(k):
            t0 = j * delta_t
            t1 = t0 + delta_t
            x0 = a[i][0] + b[i][0] * t0 + c[i][0] * t0 ** 2 + d[i][0] * t0 ** 3
            x1 = a[i][0] + b[i][0] * t1 + c[i][0] * t1 ** 2 + d[i][0] * t1 ** 3
            y0 = a[i][1] + b[i][1] * t0 + c[i][1] * t0 ** 2 + d[i][1] * t0 ** 3
            y1 = a[i][1] + b[i][1] * t1 + c[i][1] * t1 ** 2 + d[i][1] * t1 ** 3
            gui.line(begin = (x0, y0), end = (x1, y1), radius = 2, color = 0xFF0000)

    # Draw the vertices
    for i in range(n):
        gui.circle(pos = X[i], color = 0x111111, radius = 5)

    gui.show()

import taichi as ti

ti.init(arch = ti.cpu)

max_num_vertex = 4096

num_ctrl_vertex = ti.field(dtype = ti.i32, shape = ())
num_vertex = ti.field(dtype = ti.i32, shape = ())

control_vert = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex)
vert = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex)
new_vert = ti.Vector.field(2, dtype = ti.f32, shape = max_num_vertex)

show_final = ti.field(dtype = ti.i32, shape = ())
alpha = ti.field(dtype = ti.f32, shape = ())

@ti.kernel
def init_iteration():
    # vert.fill(0)
    # new_vert.fill(0)
    num_vertex[None] = num_ctrl_vertex[None]
    for i in range(num_ctrl_vertex[None]):
        vert[i] = control_vert[i]

@ti.kernel
def chaikin_subdivision():
    for i in range(num_vertex[None]):
        v0 = vert[i] if i == 0 else vert[i - 1]
        v1 = vert[i]
        v2 = vert[i] if i == num_vertex[None] - 1 else vert[i + 1]
        new_vert[i * 2] = 1 / 4 * v0 + 3 / 4 * v1
        new_vert[i * 2 + 1] = 3 / 4 * v1 + 1 / 4 * v2

    num_vertex[None] *= 2
    for i in range(num_vertex[None]):
        vert[i] = new_vert[i]

@ti.kernel
def bspline_subdivision():
    for i in range(num_vertex[None]):
        v0 = vert[i] if i == 0 else vert[i - 1]
        v1 = vert[i]
        v2 = vert[i] if i == num_vertex[None] - 1 else vert[i + 1]
        new_vert[i * 2] = 1 / 8 * v0 + 3 / 4 * v1 + 1 / 8 * v2
        new_vert[i * 2 + 1] = 1 / 2 * v1 + 1 / 2 * v2

    num_vertex[None] *= 2
    for i in range(num_vertex[None]):
        vert[i] = new_vert[i]

@ti.kernel
def interpolation_subdivision():
    for i in range(num_vertex[None]):
        v0 = vert[i] if i == 0 else vert[i - 1]
        v1 = vert[i]
        v2 = vert[i] if i == num_vertex[None] - 1 else vert[i + 1]
        v3 = vert[i] if i >= num_vertex[None] - 2 else vert[i + 2]
        new_vert[i * 2] = v1
        new_vert[i * 2 + 1] = (v1 + v2) / 2 + alpha * ((v1 + v2) / 2 - (v0 + v3) / 2)

    num_vertex[None] *= 2
    for i in range(num_vertex[None]):
        vert[i] = new_vert[i]

gui = ti.GUI('Subdivision Test System', res = (1024, 1024), background_color = 0xDDDDDD)
#gui_alpha = gui.slider('Alpha', 0, 1)
show_final[None] = True
alpha[None] = 1 / 9

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            num_ctrl_vertex[None] = 0
        elif e.key == ti.GUI.UP: 
            show_final[None] = not show_final[None]
        elif e.key == ti.GUI.LMB:
            control_vert[num_ctrl_vertex[None]] = ti.Vector([e.pos[0], e.pos[1]])
            num_ctrl_vertex[None] += 1

    #alpha[None] = gui_alpha.value

    X = control_vert.to_numpy()
    n = num_ctrl_vertex[None]

    total_iteration = 7
    colors = [0x740408, 0x5A3D7F, 0xFF0000]

    init_iteration()
    for k in range(total_iteration):
        chaikin_subdivision()
        m = num_vertex[None]
        dX = vert.to_numpy()
        for i in range(0, m - 1):
            if k == total_iteration - 1:
                gui.line(begin = dX[i], end = dX[i + 1], radius = 2, color = colors[0])
    init_iteration()
    for k in range(total_iteration):
        bspline_subdivision()
        m = num_vertex[None]
        dX = vert.to_numpy()
        for i in range(0, m - 1):
            if k == total_iteration - 1:
                gui.line(begin = dX[i], end = dX[i + 1], radius = 2, color = colors[1])

    init_iteration()
    for k in range(total_iteration):
        interpolation_subdivision()
        m = num_vertex[None]
        dX = vert.to_numpy()
        for i in range(0, m - 1):
            if k == total_iteration - 1:
                gui.line(begin = dX[i], end = dX[i + 1], radius = 2, color = colors[2])
    
    # Draw linear interpolation
    if not show_final[None]:
        for i in range(0, n - 1):
            gui.line(begin = X[i], end = X[i + 1], radius = 2, color = 0x444444)

    # Draw the vertices
    for i in range(n):
        gui.circle(pos = X[i], color = 0x111111, radius = 5)

    gui.text('chaikin_subdivision()', (0, 1), font_size = 30, color = colors[0])
    gui.text('bspline_subdivision()', (0, 0.95), font_size = 30, color = colors[1])
    gui.text('interpolation_subdivision(), alpha = 1 / 9', (0, 0.9), font_size = 30, color = colors[2])
    gui.show()

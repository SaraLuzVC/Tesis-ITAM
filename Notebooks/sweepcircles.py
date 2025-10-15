"""
Algoritmo de barrido (sweep line) para procesar eventos asociados a
círculos y puntos (puntos de entrada, extremos de círculos e intersecciones).

Basado en ideas del artículo:
  "Fast algorithms for computing B-skeletons and their relatives"
  S. V. Rao y Asish Mukhopadhyay, 2001
  Journal of the Pattern Recognition Society

Convenciones importantes:
- IDs de círculo son 1-based: circles[cid-1] es el círculo con ID=cid.
- Estado de la línea de barrido `sweepL`:
    sweepL = {
        "x": float,                 # posición x actual del barrido
        "active": set[int],         # IDs de círculos actualmente activos en x
        "conditions": list[dict],   # bandas verticales ordenadas de abajo->arriba
        "seen_inputs": set[tuple]   # (opcional) para evitar contar un input dos veces
    }
- Cada condición/banda es un dict:
    {
      "in":   set[int],                       # círculos en los que esta banda está dentro
      "out":  set[int],                       # círculos respecto a los que está fuera
      "ineq": list[("y","<"|">",(cid,"y_low"|"y_high"))]  # desigualdades simbólicas
    }

La matriz Ac es de tamaño n x n (n = número de círculos) y registra
la pertenencia de puntos de entrada a círculos activos:
- Si el punto cae dentro de exactamente 1 círculo i: Ac[i,i] += 1
- Si cae dentro de k>=2 círculos {i,j,...} activos: se incrementan simétricamente
  todos los pares (i,j) y (j,i) para esos IDs.
"""

import math
import copy

# ————————————————————————————————————————————————————————————————
# BUCLE PRINCIPAL
# ————————————————————————————————————————————————————————————————


def algoritmo(circles, points):
    """
    Bucle principal del algoritmo:
    - Construye Q con eventos iniciales (input + extremos izquierdos).
    - Inicializa sweepL y la matriz Ac de conteos.
    - Recorre Q en orden, actualizando sweepL y Ac según el tipo de evento.

    Parámetros
    ----------
    circles : list[ ((cx,cy), r) ]
        Lista de círculos (centro, radio). IDs implícitos 1..n (1-based).
    points : list[(x,y)]
        Puntos de entrada a evaluar.

    Devuelve
    --------
    (Ac, Q, sweepL) :
        Ac  : matriz n x n con conteos de pertenencia de puntos.
        Q   : lista de eventos (posiblemente ampliada con derechos/intersecciones).
        sweepL : estado final de la línea de barrido.
    """
    n = len(circles)

    # Matriz n×n de conteos (inicialmente todo 0).
    # No es una "lista de activos"; ahora es un acumulador de pertenencias.
    Ac = [[0] * n for _ in range(n)]

    # Lista de eventos: inputs + extremos izquierdos; luego ordenados.
    Q = build_Q(circles, points)

    # Inicializa línea de barrido en x apenas a la izquierda del primer evento.
    eps = 1e-8
    x0 = Q[0][0] - eps
    sweepL = {
        "x": x0,
        "active": set(),  # IDs de círculos activos en x
        "conditions": [{"in": set(), "out": set(), "ineq": []}],
        "seen_inputs": set(),  # evita doble conteo del mismo input
    }

    # — Log de estado inicial —
    print("Antes de iniciar:")
    print_Q(Q, label="Q")
    print_sweepL(sweepL, label="Línea de barrido =")
    print("Ac (matriz):")
    for row in Ac:
        print(row)

    # — Recorre la lista de eventos Q —
    i = 0
    while i < len(Q):
        x, y, kind, idx = Q[i]

        # Mueve la línea de barrido a x (no reordena bands aquí).
        advance_sweep_to(sweepL, x, circles)

        # 1) Evento de punto de entrada
        if kind == "input":
            print(f"#####     Point ({x}, {y}) is an INPUT point     #####")
            Ac = input_point(sweepL, Ac, circles, Q[i])

        # 2) Extremo izquierdo de un círculo
        elif kind == "left":
            circle_idx = list(idx)[0]
            print(
                f"#####     Point ({x}, {y}) is a LEFT endpoint of circle {idx}     #####"
            )
            Q, sweepL = leftend_point(
                Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6
            )

        # 3) Extremo derecho de un círculo
        elif kind == "right":
            circle_idx = list(idx)[0]
            print(
                f"#####     Point ({x}, {y}) is a RIGHT endpoint of circle {idx}     #####"
            )
            Q, sweepL = rightend_point(
                Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6
            )

        # 4) Intersección entre dos círculos
        elif kind == "intersection":
            ids_set = set(idx)
            print(
                f"#####     Point ({x}, {y}) is an INTERSECTION point of circles {ids_set}     #####"
            )
            Q, sweepL = intersection_point(Q, sweepL, circles, ids_set, x, y, eps=1e-6)

        # (opcional) logging
        print_Q(Q, label="Q")
        print_sweepL(sweepL, label="Línea de barrido =")
        i += 1

    # — Log de salida —
    print("Ac (final):")
    for row in Ac:
        print(row)

    return Ac, Q, sweepL


# ————————————————————————————————————————————————————————————————
# CONSTRUCCIÓN Y ORDENAMIENTO DE EVENTOS Q
# ————————————————————————————————————————————————————————————————


def build_Q(circles, points):
    """
    Construye la lista de eventos inicial:
      1) Todos los puntos de entrada (tipo 'input')
      2) Todos los extremos izquierdos de los círculos (tipo 'left')
    Luego ordena Q según la política de order_Q.
    """
    Q = []
    # 1) Puntos de entrada (input)
    for idx, (x, y) in enumerate(points, start=1):
        Q.append((x, y, "input", {idx}))
    # 2) Extremos izquierdos (left) de cada círculo
    for idx, ((cx, cy), r) in enumerate(circles, start=1):
        Q.append((cx - r, cy, "left", {idx}))
    # 3) Ordenar Q
    return order_Q(Q, circles)


def order_Q(Q, circles, kind_priority=None):
    """
    Ordena la lista de eventos `Q` con una clave compuesta que respeta cómo
    debe avanzar la línea de barrido (izquierda→derecha) y resuelve empates
    según el tipo de evento y el radio del círculo.

    Cada elemento de Q debe ser una tupla:
        (x, y, kind, ids)
      - x, y : coordenadas del evento
      - kind : str en {"right", "left", "intersection", "input"}
      - ids  : conjunto con los IDs de círculos involucrados.
               * Para "left"/"right" se asume un singleton, p.ej. {cid}
               * Para "intersection" es un set de 2 IDs, p.ej. {i, j}
               * Para "input" puede ser cualquier cosa; no afecta el orden

    Criterios de orden:
      1) x ascendente  (procesar de izquierda a derecha)
      2) prioridad por tipo (por defecto: right < left < intersection < input)
      3) desempates:
         - si ambos son 'left' en el mismo x: primero el círculo de mayor radio
         - si ambos son 'right' en el mismo x: primero el de menor radio
         - si ambos son 'intersection' en el mismo x: se usa y y luego los IDs
         - en los demás casos: y ascendente
    """
    default_priority = {"right": 0, "left": 1, "intersection": 2, "input": 3}
    if not isinstance(kind_priority, dict):
        kind_priority = default_priority

    def radius_of_ids(ids):
        # Para 'left'/'right' se espera un singleton {cid}
        if not ids:
            return 0.0
        cid = next(iter(ids))
        return circles[cid - 1][1]

    def key(ev):
        x, y, kind, ids = ev
        pri = kind_priority.get(kind, 99)
        if kind == "left":
            r = radius_of_ids(ids)
            return (x, pri, -r, y)  # mayor radio primero
        if kind == "right":
            r = radius_of_ids(ids)
            return (x, pri, r, y)  # menor radio primero
        if kind == "intersection":
            return (x, pri, 0, y, tuple(sorted(ids)))  # desempate estable por IDs
        return (x, pri, 0, y)  # input u otros

    Q.sort(key=key)
    return Q


def event_in_Q(Q, event, tol=1e-9):
    """
    True si ya existe en Q un evento con mismo tipo e ids y con (x,y) a distancia <= tol.
    Útil para deduplicar intersecciones.
    """
    x, y, etype, eids = event
    for qx, qy, qtype, qids in Q:
        if qtype != etype:
            continue
        if qids != eids:
            continue
        if abs(qx - x) <= tol and abs(qy - y) <= tol:
            return True
    return False


def add_unique_events(Q, nuevos_eventos, tol=1e-9, keep_sorted=True):
    """
    Agrega a Q los eventos (x, y, "intersection", {i,j}) de `nuevos_eventos` que no existan ya.
    Modifica Q in-place. Regresa el número de eventos agregados.
    """
    agregados = 0
    for ev in nuevos_eventos:
        # Validación muy básica de estructura
        if not (
            isinstance(ev, (list, tuple)) and len(ev) == 4 and isinstance(ev[3], set)
        ):
            continue
        if not event_in_Q(Q, ev, tol=tol):
            Q.append(ev)
            agregados += 1
    return agregados


def print_Q(Q, label="Q"):
    """
    Imprime Q con un formato legible:
    Q:  [
        (1, 11, 'left', {6}),
        (1, 23, 'left', {3}),
        ...
    ]
    Devuelve el texto impreso (por si deseas guardarlo).
    """

    def fmt_set(s):
        s = set(s)
        if not s:
            return "set()"
        return "{" + ", ".join(str(x) for x in sorted(s)) + "}"

    def fmt_val(v):
        if isinstance(v, (set, frozenset)):
            return fmt_set(v)
        return repr(v)

    def fmt_tuple(t):
        return "(" + ", ".join(fmt_val(v) for v in t) + ")"

    lines = [f"{label}:  ["]
    for i, item in enumerate(Q):
        comma = "," if i < len(Q) - 1 else ""
        lines.append("    " + fmt_tuple(item) + comma)
    lines.append("]")

    text = "\n".join(lines)
    print(text)
    return text


# ————————————————————————————————————————————————————————————————
# UTILIDADES DE SWEEP LINE
# ————————————————————————————————————————————————————————————————


def advance_sweep_to(sweepL, x_event, circles):
    """Actualiza la posición x actual de la línea de barrido al x del evento."""
    sweepL["x"] = x_event


def remove_circle_from_L(
    L,
    circle_id,
    circles=None,
    *,
    copy_result: bool = False,
    keep_empty_base: bool = True,
    rebuild_triples: bool = False,
    eps: float = 1e-9,
):
    """
    Elimina referencias a `circle_id` del estado L y, opcionalmente, reconstruye
    toda la pila de bandas en x usando los círculos que sigan activos.

    - Si rebuild_triples=True, se recalculan bandas ordenadas por todos los
      y_low/y_high de los círculos activos a la altura x = L['x'].
    - Si no quedan bandas y keep_empty_base=True, se deja una banda vacía base.

    Devuelve L (o una copia si copy_result=True).
    """
    L2 = copy.deepcopy(L) if copy_result else L

    # 1) Quitar de activos
    if "active" in L2 and isinstance(L2["active"], set):
        L2["active"].discard(circle_id)

    # 2) Purgar menciones a circle_id en 'conditions'
    def _mentions_circle(ineq):
        try:
            return (
                isinstance(ineq, tuple)
                and len(ineq) == 3
                and isinstance(ineq[2], tuple)
                and len(ineq[2]) >= 1
                and ineq[2][0] == circle_id
            )
        except Exception:
            return False

    cleaned = []
    seen = set()

    def _ineq_key(ineq):
        # Clave estable para deduplicar desigualdades
        try:
            var, op, ref = ineq
            if isinstance(ref, tuple) and len(ref) >= 2:
                cid, tag = ref[0], ref[1]
                return ("t", var, op, int(cid), str(tag))
            return ("t", var, op, str(ref))
        except Exception:
            return ("s", str(ineq))

    for cond in L2.get("conditions", []):
        in_set = set(cond.get("in", set()))
        out_set = set(cond.get("out", set()))
        ineqs = list(cond.get("ineq", []))

        in_set.discard(circle_id)
        out_set.discard(circle_id)
        ineqs = [iq for iq in ineqs if not _mentions_circle(iq)]

        # omitir banda trivialmente vacía (se garantiza base luego si procede)
        if not in_set and not out_set and not ineqs:
            continue

        key = (
            tuple(sorted(in_set)),
            tuple(sorted(out_set)),
            tuple(sorted(_ineq_key(iq) for iq in ineqs)),
        )
        if key not in seen:
            cleaned.append({"in": in_set, "out": out_set, "ineq": ineqs})
            seen.add(key)

    L2["conditions"] = cleaned

    # 3) Reconstrucción completa (opcional) de bandas en x con círculos activos
    if rebuild_triples:
        if circles is None:
            raise ValueError(
                "remove_circle_from_L(...): `circles` es requerido con rebuild_triples=True."
            )

        x = L2.get("x", None)
        if x is None:
            raise ValueError(
                "remove_circle_from_L(...): L['x'] requerido para reconstruir bandas."
            )

        active_ids = list(sorted(L2.get("active", set())))
        boundaries = []  # (y, "low"/"high", cid)
        positions = {}  # (cid, tag) -> índice en boundaries

        def _y_band_for_circle(circ, xval):
            (cx, cy), r = circ
            dx = xval - cx
            if abs(dx) > r + eps:
                return None, None
            try:
                h = (r * r - dx * dx) ** 0.5
            except ValueError:
                h = 0.0
            return cy - h, cy + h

        # recolecta todos los límites y_low/y_high a esa x
        for cid in active_ids:
            yl, yh = _y_band_for_circle(circles[cid - 1], x)
            if yl is None or yh is None:
                continue
            boundaries.append((yl, "low", cid))
            boundaries.append((yh, "high", cid))

        # ordenados de abajo a arriba, asegurando low antes que high al empatar
        boundaries.sort(key=lambda t: (t[0], 0 if t[1] == "low" else 1))
        for idx, (_, tag, cid) in enumerate(boundaries):
            positions[(cid, tag)] = idx

        if not boundaries:
            # sin bandas: deja base si se pide
            L2["conditions"] = (
                [{"in": set(), "out": set(), "ineq": []}] if keep_empty_base else []
            )
            return L2

        # Construye N+1 intervalos entre límites ordenados
        N = len(boundaries)
        rebuilt, seen2 = [], set()
        for k in range(N + 1):
            in_set, out_set, ineqs = set(), set(), []

            for cid in active_ids:
                low_i = positions.get((cid, "low"))
                high_i = positions.get((cid, "high"))
                if low_i is None or high_i is None:
                    continue

                # Clasifica el intervalo k relativo a [low_i, high_i]
                if k <= low_i:
                    out_set.add(cid)
                    ineqs.append(("y", "<", (cid, "y_low")))
                elif k <= high_i:
                    in_set.add(cid)
                else:
                    out_set.add(cid)
                    ineqs.append(("y", ">", (cid, "y_high")))

            # omite completamente vacío
            if not in_set and not out_set and not ineqs:
                continue

            key2 = (
                tuple(sorted(in_set)),
                tuple(sorted(out_set)),
                tuple(sorted(ineqs, key=lambda z: (z[0], z[1], z[2][0], z[2][1]))),
            )
            if key2 not in seen2:
                rebuilt.append({"in": in_set, "out": out_set, "ineq": ineqs})
                seen2.add(key2)

        if not rebuilt and keep_empty_base:
            rebuilt = [{"in": set(), "out": set(), "ineq": []}]

        L2["conditions"] = rebuilt

    # 4) Garantiza base vacía si quedó sin bandas y se solicita
    if keep_empty_base and not L2["conditions"]:
        L2["conditions"] = [{"in": set(), "out": set(), "ineq": []}]

    return L2


def print_sweepL(L, label="L"):
    """
    Imprime el estado sweepL con formato legible:
    L {'x': 13, 'active': {1, 2},
    'conditions': [
        {'in': set(), 'out': {1}, 'ineq': [...]},
        ...
    ]}
    Si L es str, lo imprime tal cual con el label dado.
    """
    if isinstance(L, str):
        text = f"{label} {L}"
        print(text)
        return text

    def fmt_set(s):
        if not s:
            return "set()"
        return "{" + ", ".join(str(x) for x in sorted(s)) + "}"

    def fmt_tuple(t):  # para desigualdades
        return "(" + ", ".join(repr(x) for x in t) + ")"

    def fmt_ineq(lst):
        if not lst:
            return "[]"
        return "[" + ", ".join(fmt_tuple(t) for t in lst) + "]"

    def fmt_condition(c):
        return (
            "{"
            f"'in': {fmt_set(c.get('in', set()))}, "
            f"'out': {fmt_set(c.get('out', set()))}, "
            f"'ineq': {fmt_ineq(c.get('ineq', []))}"
            "}"
        )

    x = L.get("x")
    active = fmt_set(L.get("active", set()))
    conditions = L.get("conditions", [])

    lines = []
    lines.append(f"{label} {{'x': {x}, 'active': {active}, ")
    lines.append("'conditions': [")
    for i, cond in enumerate(conditions):
        comma = "," if i < len(conditions) - 1 else ""
        lines.append("    " + fmt_condition(cond) + comma)
    lines.append("]}")
    text = "\n".join(lines)
    print(text)
    return text


# ————————————————————————————————————————————————————————————————
# PRIMITIVAS GEOMÉTRICAS Y PREDICADOS
# ————————————————————————————————————————————————————————————————


def in_circle(x, y, cx, cy, r, tolerancia=1e-12):
    """
    Devuelve True si (x,y) está dentro o sobre el círculo (cx,cy,r)
    usando tolerancia para robustez numérica.
    """
    print(
        "Evalua el punto (",
        x,
        ",",
        y,
        ") en el círculo con centro en (",
        cx,
        ",",
        cy,
        ") y radio ",
        r,
    )
    print(
        (x - cx) ** 2 + (y - cy) ** 2,
        " <= ",
        r**2 + tolerancia,
        (x - cx) ** 2 + (y - cy) ** 2 <= r**2 + tolerancia,
    )
    return (x - cx) ** 2 + (y - cy) ** 2 <= r**2 + tolerancia


def circle_y_bounds_at_x(circle, x0):
    """
    (y_low, y_high) donde la recta vertical x = x0 corta al círculo.
    (None, None) si no hay intersección vertical.
    """
    (cx, cy), r = circle
    dx = x0 - cx
    if abs(dx) > r:
        return None, None
    h = math.sqrt(max(r * r - dx * dx, 0.0))
    return cy - h, cy + h


def satisfies(cond, x, y, circles):
    """
    Verifica si el punto (x,y) satisface:
      - pertenecer a todos los círculos en cond['in']
      - NO pertenecer a todos los círculos en cond['out']
      - cumplir todas las desigualdades de cond['ineq'] respecto a y_low/y_high a x
    """
    # 1) Debe estar dentro de todos los 'in'
    for cid in cond.get("in", set()):
        (cx, cy), r = circles[cid - 1]
        print("Reviso si SÍ está en el círculo ", cid, " el punto (", x, ",", y, ")")
        if not in_circle(x, y, cx, cy, r):
            return False

    # 2) Debe estar fuera de todos los 'out'
    for cid in cond.get("out", set()):
        (cx, cy), r = circles[cid - 1]
        print("Reviso si NO está en el círculo ", cid, " el punto (", x, ",", y, ")")
        if in_circle(x, y, cx, cy, r):
            return False

    # 3) Debe satisfacer cada inecuación simbólica respecto a y_low/y_high
    for var, op, (cid, which) in cond.get("ineq", []):
        y_low, y_high = circle_y_bounds_at_x(circles[cid - 1], x)
        print("Evaluo yhigh y ylow", y_high, y_low)
        if y_low is None:
            # si x no corta al círculo, la desigualdad no es aplicable: fallar conservadoramente
            return False
        y_ref = y_low if which == "y_low" else y_high
        if op == "<" and not (y < y_ref):
            return False
        if op == ">" and not (y > y_ref):
            return False

    return True


# ————————————————————————————————————————————————————————————————
# MANEJO DE EXTREMOS E INTERSECCIONES
# ————————————————————————————————————————————————————————————————


def leftend_point(Q, sweepL, circle_id, circles, eps=1e-6):
    """
    Inserta el círculo `circle_id` en el modelo de condiciones en x = sweepL['x'].
    Divide la banda base en: down / middle / up para ese círculo y encola posibles
    intersecciones vecinas cerca de la nueva inserción. También encola su extremo
    derecho en Q y reordena Q.
    """
    (cx, cy), r = circles[circle_id - 1]
    print("Círculo izquierdo con centro en (", cx, ",", cy, ") y radio ", r)

    # Evalúa justo a la derecha del extremo izquierdo para cortar bandas de forma estable.
    x_prime = sweepL["x"] + eps
    dx = x_prime - cx
    if abs(dx) > r + 1e-15:
        # Si en x' el círculo no existe verticalmente, no se parte condición; solo activarlo.
        sweepL.setdefault("active", set()).add(circle_id)
        return Q, sweepL

    conds = sweepL["conditions"]
    print("Conditions", sweepL)

    # Encuentra banda base que contiene (x', cy) ANTES de insertar el círculo
    k = None
    for i, cond in enumerate(conds):
        if satisfies(cond, x_prime, cy, circles):
            k = i
            break
    if k is None:
        k = len(conds) - 1  # fallback conservador

    base = conds[k]
    print("base", base)
    base_in = set(base.get("in", set()))
    base_out = set(base.get("out", set()))
    base_ineq = list(base.get("ineq", []))

    # Construye las tres bandas al insertar el círculo
    down = {
        "in": set(base_in),
        "out": set(base_out) | {circle_id},
        "ineq": base_ineq + [("y", "<", (circle_id, "y_low"))],
    }
    middle = {
        "in": set(base_in) | {circle_id},
        "out": set(base_out),
        "ineq": list(base_ineq),
    }
    up = {
        "in": set(base_in),
        "out": set(base_out) | {circle_id},
        "ineq": base_ineq + [("y", ">", (circle_id, "y_high"))],
    }

    print("down", down)
    print("middle", middle)
    print("up", up)

    # Reemplaza banda base por down/middle/up
    new_conds = conds[:k] + [down, middle, up] + conds[k + 1 :]
    sweepL["conditions"] = new_conds
    sweepL.setdefault("active", set()).add(circle_id)

    # Índices útiles (por si necesitas log o vecinos)
    kb_down, kb_mid, kb_up = k, k + 1, k + 2

    # Encola intersecciones potenciales con bandas vecinas alrededor
    ids = {circle_id}
    circles_nearby = collect_circles_from_neighbors(
        ids, kb_down, kb_up, sweepL["conditions"]
    )
    print("check intersections in circles", circles_nearby)
    Q = enqueue_intersections_from_ids(
        Q, circles, circles_nearby, sweepL=sweepL, eps=1e-9, keep_sorted=True, tol=1e-9
    )

    # Agrega el extremo derecho del círculo y reordena Q
    Q.append((cx + r, cy, "right", {circle_id}))
    order_Q(Q, circles)
    return Q, sweepL


def circle_circle_intersections(c1, c2, tol=1e-12):
    """
    Intersecciones entre dos círculos c1 y c2.
    Devuelve [] (ninguna), [(x,y)] (tangente) o [(x1,y1),(x2,y2)] (dos cortes).
    """
    (x0, y0), r0 = c1
    (x1, y1), r1 = c2
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    # Casos degenerados: separados, uno dentro del otro sin corte, o coincidentes
    if d > r0 + r1 + tol:
        return []
    if d < abs(r0 - r1) - tol:
        return []
    if d < tol and abs(r0 - r1) < tol:
        return []

    # Punto medio sobre la línea de centros
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h2 = r0 * r0 - a * a
    if h2 < -tol:
        return []
    h = math.sqrt(max(h2, 0.0))

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    if h <= tol:  # tangente
        return [(xm, ym)]

    # Dos cortes
    rx = -dy * (h / d)
    ry = dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def rightend_point(Q, sweepL, circle_id, circles, eps=1e-6):
    """
    Procesa el extremo derecho de `circle_id` en x.
    - Localiza 'middle' (banda con circle_id en 'in') que contiene (x - eps, cy)
    - Toma sus vecinas 'down' y 'up'
    - Fusiona up y down quitando referencias al círculo
    - Actualiza sweepL (y reconstruye bandas en x si se pide)
    - NO agrega evento 'right' adicional a Q (ya estaba en Q desde left)
    """
    (cx, cy), r = circles[circle_id - 1]
    print("Círculo derecho con centro en (", cx, ",", cy, ") y radio ", r)

    x_prime = sweepL["x"] - eps  # evalúa justo antes del extremo derecho
    conds = sweepL["conditions"]

    # Localiza la banda 'middle' adecuada
    k = None
    for i, cond in enumerate(conds):
        if circle_id in cond.get("in", set()) and satisfies(cond, x_prime, cy, circles):
            k = i
            break

    if k is None:
        sweepL.setdefault("active", set()).discard(circle_id)
        print(
            "No se encontró región con el círculo en 'in' para (x', cy); no se fusiona."
        )
        return Q, sweepL

    # Deben existir vecinos arriba y abajo
    if k - 1 < 0 or k + 1 >= len(conds):
        sweepL.setdefault("active", set()).discard(circle_id)
        print("No hay vecinos up/down contiguos; no se fusiona.")
        return Q, sweepL

    down, mid, up = conds[k - 1], conds[k], conds[k + 1]
    print("down", down)
    print("middle", mid)
    print("up", up)

    # Fusiona up y down, purgando desigualdades del círculo que cierra
    def _sin_ineq_del_circulo(ineqs, cid):
        filtradas = []
        for t in ineqs:
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                filtradas.append(t)
                continue
            var, op, ref = t
            if var == "y" and isinstance(ref, tuple) and ref[0] == cid:
                continue
            filtradas.append(t)
        return filtradas

    base_in = set(down.get("in", set())) & set(up.get("in", set()))
    base_out = (set(down.get("out", set())) & set(up.get("out", set()))) - {circle_id}
    ineq_down = _sin_ineq_del_circulo(down.get("ineq", []), circle_id)
    ineq_up = _sin_ineq_del_circulo(up.get("ineq", []), circle_id)

    base_ineq = []
    for t in ineq_down + ineq_up:
        if t not in base_ineq:
            base_ineq.append(t)

    merged = {"in": base_in, "out": base_out, "ineq": base_ineq}
    print("merged", merged)

    # Reemplaza [down, middle, up] por [merged]
    sweepL["conditions"] = conds[: k - 1] + [merged] + conds[k + 2 :]
    sweepL.setdefault("active", set()).discard(circle_id)

    # Opcional: reconstrucción completa de bandas en x
    sweepL = remove_circle_from_L(
        sweepL,
        circle_id,
        circles=circles,
        rebuild_triples=True,  # reconstruye la partición vertical entera en x
        keep_empty_base=True,
    )
    return Q, sweepL


# ————————————————————————————————————————————————————————————————
# REGISTRO DE INPUTS EN LA MATRIZ Ac
# ————————————————————————————————————————————————————————————————


def input_point(sweepL, Ac, circles, q, eps=1e-9):
    """
    Registra el punto de entrada en la matriz Ac usando SOLO los círculos activos:
    - Dentro de 1 círculo i -> Ac[i,i] += 1
    - Dentro de k>=2 -> incrementa todos los pares (i,j) y (j,i) con i != j.

    Evita doble conteo del mismo punto usando sweepL['seen_inputs'].
    """
    # Solo tomamos x,y del evento q (puede venir con más campos)
    if not (isinstance(q, tuple) and len(q) >= 2):
        return Ac
    x, y = q[0], q[1]

    # Evita contar dos veces el mismo input exacto (según redondeo)
    key = (round(x, 9), round(y, 9))
    seen = sweepL.setdefault("seen_inputs", set())
    if key in seen:
        return Ac
    seen.add(key)

    active_ids = set(sweepL.get("active", set()))
    if not active_ids:
        return Ac

    def point_in_circle(px, py, circle):
        (cx, cy), r = circle
        return (px - cx) ** 2 + (py - cy) ** 2 <= r**2 + eps  # incluye borde

    # ¿Qué círculos ACTIVOS contienen el punto?
    containing = []
    for cid in sorted(active_ids):
        if 1 <= cid <= len(circles) and point_in_circle(x, y, circles[cid - 1]):
            containing.append(cid)

    if not containing:
        return Ac

    if len(containing) == 1:
        i = containing[0] - 1
        Ac[i][i] += 1
    else:
        # incrementa simétricamente todos los pares (i,j), j>i
        for a in range(len(containing)):
            for b in range(a + 1, len(containing)):
                i, j = containing[a] - 1, containing[b] - 1
                Ac[i][j] += 1
                Ac[j][i] += 1
    return Ac


# ————————————————————————————————————————————————————————————————
# INTERSECCIONES: LOCALIZACIÓN Y RECONSTRUCCIÓN DE BANDAS
# ————————————————————————————————————————————————————————————————


def intersection_point(Q, sweepL, circles, ids, x, y, eps=1e-6):
    """
    Maneja un evento de intersección (x,y) entre los dos círculos en `ids`.
    - Localiza las bandas 'down' y 'up' justo antes de la intersección.
    - Determina la 'middle' entre ellas.
    - Reconstruye la banda del medio tras la intersección (flip de pertenencia).
    - Encola intersecciones vecinas potenciales y reordena Q.
    """
    assert (
        isinstance(ids, (set, frozenset)) and len(ids) == 2
    ), "ids must be a set of two circle IDs"

    conds = sweepL["conditions"]

    # Subconjunto relevante de condiciones para estos ids (no se usa íntegralmente aquí)
    conds_sel, idx_sel = pick_conditions_for_ids(ids, sweepL)
    subset = set(ids)

    # Sondas verticales y nudges para clasificar bandas de forma robusta
    dy = max(1e-8, eps)
    x_before, x_after, dy, y_up, y_up2 = adaptive_probes_for_intersection(
        x, y, ids, circles
    )

    print("REGIONES ANTES")
    print("*********DOWN")
    kb_down = find_condition_index_at_from_subset_safe(
        x_before, y - dy, circles, conds, subset
    )
    print(
        "before down",
        kb_down,
        (
            conds[kb_down]
            if kb_down is not None and 0 <= kb_down < len(conds)
            else "(no exact match)"
        ),
    )

    print("*******UP")
    kb_up = find_condition_index_at_from_subset_safe(
        x_before, y_up, circles, conds, subset
    )
    print(
        "before up",
        kb_up,
        (
            conds[kb_up]
            if kb_up is not None and 0 <= kb_up < len(conds)
            else "(no exact match)"
        ),
    )

    print("************MIDDLE")
    kb_middle = pick_middle_index_between(conds, kb_down, kb_up, ids)
    if kb_middle is None:
        # fallback: usar segunda sonda hacia arriba o y exacta
        kb_middle = find_condition_index_at_from_subset_safe(
            x_before, y_up2, circles, conds, subset
        )
        if kb_middle is None:
            kb_middle = find_condition_index_at_from_subset_safe(
                x_before, y, circles, conds, subset
            )
    print("before middle", kb_middle, conds[kb_middle])

    print("REGIONES DESPUES")
    # Por definición: up y down se conservan; solo se reconstruye middle
    ka_down = kb_down
    ka_up = kb_up

    # Reconstruye banda central post-intersección por "flip" de pertenencia de ids
    middle_after = rebuild_middle_after_by_flip(
        conds, kb_middle, ids, x_after, y, circles, eps
    )
    conds[kb_middle] = middle_after
    print(middle_after)

    # Encolar intersecciones derivadas en vecinos y reordenar Q
    circles_nearby = collect_circles_from_neighbors(
        ids, kb_down, kb_up, sweepL["conditions"]
    )
    print("check intersections in circles", circles_nearby)
    Q = enqueue_intersections_from_ids(
        Q, circles, circles_nearby, sweepL=sweepL, eps=1e-9, keep_sorted=True, tol=1e-9
    )
    order_Q(Q, circles)

    return Q, sweepL


def rebuild_middle_after_by_flip(conds, kb_middle, ids, x_after, y, circles, eps=1e-9):
    """
    Reconstruye la banda 'middle' tras la intersección, cambiando la pertenencia
    de ambos círculos en `ids`:
      - Si estaban en 'in', pasan a 'out' con una inecuación '< y_low' o '> y_high'
        según el lado más cercano de la banda.
      - Si estaban en 'out', pasan a 'in'.
      - Mantiene el resto (otros círculos/ineqs) sin cambios.
    """
    base = conds[kb_middle]

    # Purga referencias previas a ids
    outc = {
        "in": set(c for c in base.get("in", set()) if c not in ids),
        "out": set(c for c in base.get("out", set()) if c not in ids),
        "ineq": [
            (v, op, arg)
            for (v, op, arg) in base.get("ineq", [])
            if not (isinstance(arg, (tuple, list)) and len(arg) == 2 and arg[0] in ids)
        ],
    }

    for cid in sorted(ids):
        was_in = cid in base.get("in", set())
        if was_in:
            # ahora pasa a 'out' — decide si va con '< y_low' o '> y_high'
            (cx, cy), r = circles[cid - 1]
            yl, yh = _circle_y_band_at_x(circles[cid - 1], x_after)
            if yl is None:
                outc["out"].add(cid)
            else:
                mid = 0.5 * (yl + yh)
                if y <= mid:
                    outc["out"].add(cid)
                    outc["ineq"].append(("y", "<", (cid, "y_low")))
                else:
                    outc["out"].add(cid)
                    outc["ineq"].append(("y", ">", (cid, "y_high")))
        else:
            # estaba fuera -> pasa a dentro
            outc["in"].add(cid)

    return outc


# ————————————————————————————————————————————————————————————————
# SELECCIÓN / BÚSQUEDA DE BANDAS
# ————————————————————————————————————————————————————————————————


def pick_conditions_for_ids(ids, sweepL):
    """
    Dada la dupla de IDs `ids={a,b}`, intenta seleccionar un pequeño subconjunto
    de bandas relevantes del stack sweepL['conditions'] (por ejemplo, las que
    contienen a ambos en 'in', o la más cercana a cada uno) para acelerar
    búsquedas/diagnósticos.
    Devuelve (conds_sel, idx_sel) manteniendo el orden vertical.
    """
    assert (
        isinstance(ids, (set, frozenset)) and len(ids) == 2
    ), "ids debe tener exactamente dos elementos"
    a, b = sorted(ids)
    conds = sweepL.get("conditions", [])

    # 1) Bandas donde 'in' contiene a ambos
    both_idxs = [i for i, c in enumerate(conds) if ids.issubset(c.get("in", set()))]
    if both_idxs:
        start, end = min(both_idxs), max(both_idxs)
        lo = max(0, start - 2)
        hi = min(len(conds) - 1, end + 2)
        idx_sel = list(range(lo, hi + 1))
        return [conds[i] for i in idx_sel], idx_sel

    # 2) De lo contrario, buscar índices individuales
    idxs_a = [i for i, c in enumerate(conds) if a in c.get("in", set())]
    idxs_b = [i for i, c in enumerate(conds) if b in c.get("in", set())]

    if not idxs_a and not idxs_b:
        return [], []
    if not idxs_a or not idxs_b:
        base = (idxs_a or idxs_b)[0]
        lo = max(0, base - 1)
        hi = min(len(conds) - 1, base + 1)
        idx_sel = list(range(lo, hi + 1))
        return [conds[i] for i in idx_sel], idx_sel

    # Par (i,j) con mínima distancia vertical
    best = None
    for i in idxs_a:
        for j in idxs_b:
            dist = abs(i - j)
            if best is None or dist < best[0]:
                best = (dist, i, j)

    _, i, j = best
    low, high = (i, j) if i < j else (j, i)

    if high - low >= 2:
        middle = low + 1
        idx_sel = [low, middle, high]
    else:
        idx_sel = [low, high]
        neighbor = None
        if low - 1 >= 0:
            neighbor = low - 1
        elif high + 1 < len(conds):
            neighbor = high + 1
        if neighbor is not None:
            idx_sel = sorted(set(idx_sel + [neighbor]))

    return [conds[k] for k in idx_sel], idx_sel


def find_condition_index_at_from_subset_safe(x, y, circles, conds, subset, tol=1e-12):
    """
    Encuentra el índice de la primera banda en `conds` que sea consistente con:
      - pertenencias 'in'/'out' SOLO respecto a ids en `subset`
      - todas sus desigualdades 'ineq' evaluadas en (x,y)
    Si no encuentra match exacto con subset, cae a un 'scoring' usando todos los ids
    mencionados en conds (menor penalización gana).
    """
    # 1) match exacto con subset
    ins_sub, out_sub = _classify_inside_outside(x, y, circles, subset, tol)
    candidates = []
    for i, c in enumerate(conds):
        if not (c["in"] & subset <= ins_sub):  # requiere estar dentro y no lo está
            continue
        if not (c["out"] & subset <= out_sub):  # requiere estar fuera y no lo está
            continue
        ok = True
        for _var, op, (cid, which) in c.get("ineq", []):
            if not _ineq_holds(y, op, cid, which, x, circles, tol):
                ok = False
                break
        if ok:
            candidates.append(i)
    if candidates:
        return candidates[0]

    # 2) fallback: usa TODOS los ids que aparecen en alguna banda e intenta el mejor puntaje
    all_ids = set()
    for c in conds:
        all_ids |= (
            c["in"] | c["out"] | {cid for (_v, _op, (cid, _w)) in c.get("ineq", [])}
        )
    ins_all, out_all = _classify_inside_outside(x, y, circles, all_ids, tol)

    best_i, best_score = None, float("inf")
    for i, c in enumerate(conds):
        score = 0
        score += len(c["in"] - ins_all)
        score += len(c["out"] - out_all)
        for _v, op, (cid, which) in c.get("ineq", []):
            yl, yh = _circle_y_band(circles, cid, x)
            if yl is None:
                score += 2
                continue
            target = yl if which == "y_low" else yh
            ok = _ineq_holds(y, op, cid, which, x, circles, tol)
            if not ok:
                score += 1 + abs(y - target)
        if score < best_score:
            best_score, best_i = score, i

    return best_i


import math  # (redundante pero inofensivo en tu script original)


def _circle_y_band(circles, cid, x, tol=1e-15):
    """(y_low, y_high) del círculo cid a la vertical x; (None,None) si no corta."""
    (cx, cy), r = circles[cid - 1]
    dx = x - cx
    d2 = r * r - dx * dx
    if d2 < tol:
        if d2 < -tol:
            return (None, None)
        d2 = 0.0
    h = math.sqrt(d2)
    return (cy - h, cy + h)


def _ineq_holds(y, op, cid, which, x, circles, tol=1e-12):
    """Evalúa 'y < y_low(cid)' o 'y > y_high(cid)' (o sus variantes <=, >=) en (x,y)."""
    yl, yh = _circle_y_band(circles, cid, x)
    if yl is None:
        return False
    target = yl if which == "y_low" else yh
    if op == "<":
        return y < target - tol
    if op == ">":
        return y > target + tol
    if op == "<=":
        return y <= target + tol
    if op == ">=":
        return y >= target - tol
    return False


def _classify_inside_outside(x, y, circles, ids, tol=1e-12):
    """Clasifica ids en (inside_set, outside_set) según (x,y)."""
    inside, outside = set(), set()
    for cid in ids:
        (cx, cy), r = circles[cid - 1]
        val = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        (inside if val <= tol else outside).add(cid)
    return inside, outside


def _circle_y_band_at_x(circle, x, tol=1e-12):
    """(y_low, y_high) del círculo en la vertical x; (None,None) si |x-cx|>r."""
    (cx, cy), r = circle
    dx = x - cx
    if abs(dx) > r + tol:
        return (None, None)
    h2 = r * r - dx * dx
    if h2 < -tol:
        return (None, None)
    h = math.sqrt(max(h2, 0.0))
    return (cy - h, cy + h)


def _purge_ids_from_condition(cond, ids):
    """Quita cualquier mención a `ids` en in/out/ineq de la condición copiada."""
    newc = {"in": set(), "out": set(), "ineq": []}
    newc["in"] = {c for c in cond.get("in", set()) if c not in ids}
    newc["out"] = {c for c in cond.get("out", set()) if c not in ids}
    for item in cond.get("ineq", []):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and isinstance(item[2], (list, tuple))
            and len(item[2]) == 2
        ):
            cid = item[2][0]
            if cid in ids:
                continue
        newc["ineq"].append(item)
    return newc


def _circle_ids_from_condition(cond):
    """IDs de todos los círculos mencionados en in/out/ineq de una condición."""
    s = set()
    s |= set(cond.get("in", set()))
    s |= set(cond.get("out", set()))
    for item in cond.get("ineq", []):
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and isinstance(item[2], (list, tuple))
            and len(item[2]) >= 1
        ):
            cid = item[2][0]
            if isinstance(cid, int):
                s.add(cid)
    return s


def collect_circles_from_neighbors(ids, kb_down, kb_up, conds):
    """
    Desde conds[kb_down] (y su vecina inferior) y conds[kb_up] (y su vecina superior),
    recolecta todos los IDs de círculo referenciados (in/out/ineq). Incluye además `ids`.
    Retorna una lista única preservando orden de primera aparición.
    """
    n = len(conds)
    order = []

    def add_from_index(idx):
        if idx is None:
            return
        if 0 <= idx < n:
            for cid in _circle_ids_from_condition(conds[idx]):
                if cid not in order:
                    order.append(cid)

    add_from_index(kb_down)
    if isinstance(kb_down, int):
        add_from_index(kb_down - 1)
    add_from_index(kb_up)
    if isinstance(kb_up, int):
        add_from_index(kb_up + 1)

    for cid in ids:
        if cid not in order:
            order.append(cid)

    return order


def intersections_from_ids(ids_list, circles):
    """
    Todas las intersecciones por pares de una lista de IDs 1-based.
    Devuelve eventos Q tipo: (x, y, 'intersection', {i, j}).
    """
    uniq = []
    for cid in ids_list:
        if isinstance(cid, int) and 1 <= cid <= len(circles) and cid not in uniq:
            uniq.append(cid)

    events = []
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            i, j = uniq[a], uniq[b]
            pts = circle_circle_intersections(circles[i - 1], circles[j - 1])
            for x, y in pts:
                events.append((x, y, "intersection", {i, j}))
    return events


def enqueue_intersections_from_ids(
    Q, circles, ids_list, sweepL=None, eps=1e-9, keep_sorted=True, tol=1e-9
):
    """
    Construye eventos de intersección a partir de ids_list y los agrega a Q.
    - Si sweepL se da: solo conserva eventos estrictamente adelante (x > sweepL['x'] + eps)
    - Deduplica con tolerancia `tol`
    - Reordena Q al final (si procede)
    """
    new_events = intersections_from_ids(ids_list, circles)

    if sweepL is not None and "x" in sweepL:
        cutoff = sweepL["x"] + eps
        new_events = [ev for ev in new_events if ev[0] > cutoff]

    try:
        add_unique_events(Q, new_events, tol=tol, keep_sorted=keep_sorted)
    except NameError:
        # Fallback defensivo (tu código ya define add_unique_events)
        def _almost(a, b):
            return abs(a - b) <= tol

        def _same_event(e1, e2):
            x1, y1, k1, s1 = e1
            x2, y2, k2, s2 = e2
            return (
                k1 == k2 == "intersection"
                and s1 == s2
                and _almost(x1, x2)
                and _almost(y1, y2)
            )

        for ev in new_events:
            if not any(_same_event(ev, old) for old in Q):
                Q.append(ev)
        try:
            order_Q(Q, circles)
        except NameError:
            Q.sort(key=lambda e: (e[0], e[1]))

    try:
        order_Q(Q, circles)
    except NameError:
        Q.sort(key=lambda e: (e[0], e[1]))

    return Q


def _cond_class_vs_ids(cond, ids):
    """Clasifica una banda vs {i,j}: 'inside_both', 'between', 'one_inside' u 'other'."""
    ids = set(ids)
    cin = set(cond.get("in", set())) & ids
    cout = set(cond.get("out", set())) & ids
    ineq = cond.get("ineq", [])

    if cin == ids:
        return "inside_both"

    if not cin and cout == ids:
        lt = any(
            v == "y" and op == "<" and (rhs[0] in ids)
            for (v, op, rhs) in ineq
            if isinstance(rhs, tuple)
        )
        gt = any(
            v == "y" and op == ">" and (rhs[0] in ids)
            for (v, op, rhs) in ineq
            if isinstance(rhs, tuple)
        )
        if lt and gt:
            return "between"

    if len(cin) == 1 and len(ids - cin) == 1:
        return "one_inside"

    return "other"


def pick_middle_index_between(conds, kb_down, kb_up, ids):
    """
    Elige el índice de la verdadera banda 'middle' entre kb_down y kb_up.
    Preferencias: si encuentra explícitamente 'inside_both' o 'between', lo usa.
    Si no, toma un 'one_inside' válido o, en última instancia, el slot intermedio.
    """
    lo = min(kb_down, kb_up)
    hi = max(kb_down, kb_up)
    if hi - lo < 2:
        return None  # nada entre ambos

    best_one_inside = None
    for i in range(lo + 1, hi):
        typ = _cond_class_vs_ids(conds[i], ids)
        if typ in ("inside_both", "between"):
            return i
        if typ == "one_inside" and best_one_inside is None:
            best_one_inside = i

    if best_one_inside is not None:
        return best_one_inside
    return lo + 1


def _safe_circle_band_at_x(circle, x, tol=1e-15):
    """
    Variante defensiva de _circle_y_band_at_x: recorta pequeños negativos de precisión
    y devuelve (None,None) si no hay corte vertical.
    """
    (cx, cy), r = circle
    dx = x - cx
    d2 = r * r - dx * dx
    if d2 < tol:
        if d2 < -tol:
            return (None, None)
        d2 = 0.0
    h = math.sqrt(d2)
    return (cy - h, cy + h)


def adaptive_probes_for_intersection(
    x,
    y,
    ids,
    circles,
    x_eps_min=1e-9,
    x_eps_scale=1e-12,
    frac_band=0.10,
    frac_clear=0.45,
    floor_mult=10.0,
):
    """
    Calcula desplazamientos robustos alrededor de una intersección:
      - epsx horizontal (depende del tamaño del problema)
      - dy vertical hacia arriba que se mantenga dentro de todas las bandas
        de los círculos en `ids` tanto en x-epsx como en x+epsx.

    Devuelve: (x_before, x_after, dy, y_up, y_up2)
    """
    # epsx escala con el tamaño del problema (x y radios)
    max_r = max(circles[cid - 1][1] for cid in ids)
    epsx = max(x_eps_min, x_eps_scale * (abs(x) + max_r))

    x_before = x - epsx
    x_after = x + epsx

    bands, clears = [], []
    for cid in sorted(ids):
        yl, yh = _safe_circle_band_at_x(circles[cid - 1], x_before)
        if yl is None:
            continue
        bands.append(yh - yl)
        c_low = max(0.0, y - yl)
        c_high = max(0.0, yh - y)
        clears.append(min(c_low, c_high))

    if not bands:
        bands = [1.0]
    if not clears:
        clears = [1.0]

    dy_raw = frac_band * min(bands)
    dy_clear = frac_clear * min(clears)
    dy_floor = floor_mult * epsx
    dy = max(dy_floor, min(dy_raw, dy_clear))

    y_up = y + dy
    y_up2 = y + min(2.0 * dy, 0.9 * min(clears))  # deja margen del arco

    return x_before, x_after, dy, y_up, y_up2

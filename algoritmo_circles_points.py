import math
from itertools import combinations


def build_Q(circles, points):
    Q = []
    # 1) Input points
    for idx, (x, y) in enumerate(points, start=1):
        Q.append((x, y, "input", {idx}))
    # 2) Left endpoints of circles
    for idx, ((cx, cy), r) in enumerate(circles, start=1):
        Q.append((cx - r, cy, "left", {idx}))
    # 3) Order the list Q
    return order_Q(Q, circles)


def order_Q(Q, circles, kind_priority=None):
    """
    Sort Q by:
      1) x ascending
      2) kind priority (default: right < left < intersection < input)
      3) tie-breaks:
         - 'left'  at same x:   larger radius FIRST
         - 'right' at same x:   smaller radius FIRST
         - otherwise: y ascending
    """
    default_priority = {"right": 0, "left": 1, "intersection": 2, "input": 3}
    if not isinstance(kind_priority, dict):
        kind_priority = default_priority

    def radius_of_ids(ids):
        # ids is a singleton set like {cid}
        if not ids:
            return 0.0
        cid = next(iter(ids))
        return circles[cid - 1][1]

    def key(ev):
        x, y, kind, ids = ev
        pri = kind_priority.get(kind, 99)
        if kind == "left":
            r = radius_of_ids(ids)
            return (x, pri, -r, y)  # larger r first
        if kind == "right":
            r = radius_of_ids(ids)
            return (x, pri, r, y)  # smaller r first
        if kind == "intersection":
            return (x, pri, 0, y, tuple(sorted(ids)))
        return (x, pri, 0, y)

    Q.sort(key=key)
    return Q


def event_in_Q(Q, event, tol=1e-9):
    """
    Regresa True si existe en Q un evento con mismo tipo e ids y con (x,y)
    a distancia <= tol. De lo contrario False.
    event debe ser (x, y, tipo, ids_set)
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
    Agrega a Q los eventos de `nuevos_eventos` (tuplas estilo Q: (x, y, "intersection", {i,j}))
    que no existan ya en Q (según `event_in_Q`).
    Modifica Q in place. Regresa el número de eventos agregados.
    """
    agregados = 0
    for ev in nuevos_eventos:
        # Validación básica de estructura
        if not (
            isinstance(ev, (list, tuple)) and len(ev) == 4 and isinstance(ev[3], set)
        ):
            continue
        if not event_in_Q(Q, ev, tol=tol):
            Q.append(ev)
            agregados += 1

    # if agregados and keep_sorted:
    #     # Ordena por (x, y) como tu order_Q
    #     Q.sort(key=lambda t: (t[0], t[1]))
    return agregados


def print_Q(Q, label="Q"):
    """
    Imprime Q con el formato:

    Q:  [
        (1, 11, 'left', {6}),
        (1, 23, 'left', {3}),
        ...
    ]

    Devuelve el texto por si quieres guardarlo o registrarlo.
    """

    def fmt_set(s):
        # Acepta set/frozenset/iterable; garantiza orden y el caso set()
        s = set(s)
        if not s:
            return "set()"
        return "{" + ", ".join(str(x) for x in sorted(s)) + "}"

    def fmt_val(v):
        if isinstance(v, (set, frozenset)):
            return fmt_set(v)
        # repr conserva comillas para strings y decimales como 13.0 si vienen así
        return repr(v)

    def fmt_tuple(t):
        return "(" + ", ".join(fmt_val(v) for v in t) + ")"

    lines = [f"{label}:  ["]  # Ojo: dos espacios tras los dos puntos
    for i, item in enumerate(Q):
        comma = "," if i < len(Q) - 1 else ""
        lines.append("    " + fmt_tuple(item) + comma)
    lines.append("]")

    text = "\n".join(lines)
    print(text)
    return text


def advance_sweep_to(sweepL, x_event, circles):
    sweepL["x"] = x_event

import copy


def _mentions(ineq, cid, op=None):
    # ineq like ('y','<', (cid, 'y_low')) or ('y','>', (cid,'y_high'))
    if not (
        isinstance(ineq, tuple)
        and len(ineq) == 3
        and ineq[0] == "y"
        and isinstance(ineq[2], tuple)
    ):
        return False
    tgt_cid, tgt_tag = ineq[2]
    if tgt_cid != cid:
        return False
    if op is None:
        return True
    return op == ineq[1]


def _has_guard(conds, cid, op):  # op '<' for y_low, '>' for y_high
    for c in conds:
        for t in c.get("ineq", ()):
            if _mentions(t, cid, op):
                return True
    return False


def _first_middle_idx(conds, cid):
    for i, c in enumerate(conds):
        if cid in c.get("in", set()):
            return i
    return None


def _dedup_adjacent(conds):
    out = []
    for c in conds:
        if not out or out[-1] != c:
            out.append(c)
    return out

import copy


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
    Purge every reference to `circle_id` from the sweep-line state L and,
    optionally, rebuild the vertical bands at L['x'] from the remaining active circles.

    Params
    ------
    L : dict with keys {'x', 'active', 'conditions'}
    circle_id : int
    circles : list[((cx,cy), r)] or None
        Required if rebuild_triples=True.
    copy_result : bool
        If True, work on a deep copy; else mutate in place.
    keep_empty_base : bool
        Keep at least one empty base condition if nothing else remains.
    rebuild_triples : bool
        If True, rebuild the entire condition stack at x from remaining L['active'].
    eps : float
        Numerical tolerance when computing y-bands.
    """
    L2 = copy.deepcopy(L) if copy_result else L

    # 1) remove from active
    if "active" in L2 and isinstance(L2["active"], set):
        L2["active"].discard(circle_id)

    # 2) purge references to circle_id from conditions
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

        # skip fully empty; we'll ensure a base band later if needed
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

    # 3) Optionally rebuild the *entire* stack from surviving active circles
    if rebuild_triples:
        if circles is None:
            raise ValueError(
                "remove_circle_from_L(...): `circles` is required when rebuild_triples=True."
            )

        x = L2.get("x", None)
        if x is None:
            raise ValueError(
                "remove_circle_from_L(...): L['x'] is required to rebuild bands."
            )

        active_ids = list(sorted(L2.get("active", set())))
        # compute all vertical boundaries at x for active circles
        boundaries = []  # list of (y, tag, cid)
        positions = {}  # map (cid,'low'/'high') -> index in boundaries

        def _y_band_for_circle(circ, xval):
            (cx, cy), r = circ
            dx = xval - cx
            if abs(dx) > r + eps:
                return None, None
            # y offset
            try:
                h = (r * r - dx * dx) ** 0.5
            except ValueError:
                h = 0.0
            return cy - h, cy + h

        for cid in active_ids:
            yl, yh = _y_band_for_circle(circles[cid - 1], x)
            if yl is None or yh is None:
                continue
            boundaries.append((yl, "low", cid))
            boundaries.append((yh, "high", cid))

        boundaries.sort(key=lambda t: (t[0], 0 if t[1] == "low" else 1))
        for idx, (_, tag, cid) in enumerate(boundaries):
            positions[(cid, tag)] = idx

        # If no boundaries (no active vertical spans), keep a base band if requested
        if not boundaries:
            L2["conditions"] = (
                [{"in": set(), "out": set(), "ineq": []}] if keep_empty_base else []
            )
            return L2

        # Build N+1 intervals from the ordered boundaries
        N = len(boundaries)
        rebuilt = []
        seen2 = set()
        for k in range(N + 1):
            in_set, out_set, ineqs = set(), set(), []

            for cid in active_ids:
                low_i = positions.get((cid, "low"))
                high_i = positions.get((cid, "high"))
                if low_i is None or high_i is None:
                    # circle not present at this x (skip)
                    continue

                # classify interval k relative to [low_i, high_i]
                if k <= low_i:
                    # below y_low
                    out_set.add(cid)
                    ineqs.append(("y", "<", (cid, "y_low")))
                elif k <= high_i:
                    # between y_low and y_high: inside
                    in_set.add(cid)
                else:
                    # above y_high
                    out_set.add(cid)
                    ineqs.append(("y", ">", (cid, "y_high")))

            # drop trivially empty only if you don't want fully empty bands
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

        # If the rebuild produced nothing, keep an empty base if requested
        if not rebuilt and keep_empty_base:
            rebuilt = [{"in": set(), "out": set(), "ineq": []}]

        L2["conditions"] = rebuilt

    # 4) Ensure at least one empty base if nothing remains
    if keep_empty_base and not L2["conditions"]:
        L2["conditions"] = [{"in": set(), "out": set(), "ineq": []}]

    return L2


def print_sweepL(L, label="L"):
    """
    Imprime sweepL con el formato:
    L {'x': 13, 'active': {1, 2},
    'conditions': [
        {'in': set(), 'out': {1}, 'ineq': [...]},
        ...
    ]}
    Si L es str, solo la antepone con el label y la imprime.
    Devuelve el texto final por si quieres guardarlo o loguearlo.
    """
    if isinstance(L, str):
        text = f"{label} {L}"
        print(text)
        return text

    def fmt_set(s):
        if not s:
            return "set()"
        return "{" + ", ".join(str(x) for x in sorted(s)) + "}"

    def fmt_tuple(t):
        # repr para strings; números tal cual
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


def in_circle(x, y, cx, cy, r, tolerancia=1e-12):
    """
    Evalua si un punto se encuentra dentro de un círculo.
    Regresa Verdadero o Falso, dependiendo si cumpe o no.
    Se usa una tolerancia para evitar errores por punto flotante
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
    Regresa los cortes (y_low, y_high) de la línea x = x0
    con el círculo con centro en (cx, cy) y radio r
    Si la línea no intersecta al círculo regresa (None, None).
    """
    (cx, cy), r = circle
    # Calcula la distancia horizontal entre el centro del círculo
    # y el punto x0
    dx = x0 - cx
    # Si esa distancia es mayor al radio, no se intersectan
    if abs(dx) > r:
        return None, None
    # Si s[i se intersectan, calcula las intersecciones
    h = math.sqrt(max(r * r - dx * dx, 0.0))
    return cy - h, cy + h


# Hay 2 tipos de condiciones que puede tener una región
# con respecto a un cículo activo en la línea de barrido sweepL:
#    * Que esté dentro de ese círculo activo en sweepL
#    * Que esté fuera de ese círculo activo en sweepL
# y, si está fuera puede estar:
#    * Por encima de ese círculo
#    * Por debajo de ese círculo


def satisfies(cond, x, y, circles):
    # para evaluar si está dentro del círculo
    for cid in cond.get("in", set()):
        (cx, cy), r = circles[cid - 1]
        print("Reviso si SÍ está en el círculo ", cid, " el punto (", x, ",", y, ")")
        if not in_circle(x, y, cx, cy, r):
            return False

    # para evaluar si está fuera del circulo
    for cid in cond.get("out", set()):
        (cx, cy), r = circles[cid - 1]
        print("Reviso si NO está en el círculo ", cid, " el punto (", x, ",", y, ")")
        if in_circle(x, y, cx, cy, r):
            return False

    # para evaluar si está por encima o por debajo
    for var, op, (cid, which) in cond.get("ineq", []):
        y_low, y_high = circle_y_bounds_at_x(circles[cid - 1], x)
        print("Evaluo yhigh y ylow")
        if y_low is None:
            # If the sweep line x doesn't intersect this circle at all,
            # this inequality is not meaningful. Be conservative and fail.
            return False
        y_ref = y_low if which == "y_low" else y_high
        if op == "<" and not (y < y_ref):
            return False
        if op == ">" and not (y > y_ref):
            return False

    return True


def leftend_point(Q, sweepL, circle_id, circles, eps=1e-6):
    """
    Insert circle `circle_id` into the conditions model at sweepL['x'].
    Ahora usa vecinos (kb_down/kb_up) para encolar intersecciones cercanas,
    igual que intersection_point.
    """
    (cx, cy), r = circles[circle_id - 1]
    print("Círculo izquierdo con centro en (", cx, ",", cy, ") y radio ", r)

    # Evaluar justo a la derecha del extremo izquierdo
    x_prime = sweepL["x"] + eps
    dx = x_prime - cx
    if abs(dx) > r + 1e-15:
        # Fuera del span vertical en x' -> no partimos condición
        sweepL.setdefault("active", set()).add(circle_id)
        return Q, sweepL

    conds = sweepL["conditions"]
    print("Conditions", sweepL)

    # Buscar la condición base que contiene (x', cy) ANTES de insertar el círculo
    k = None
    for i, cond in enumerate(conds):
        if satisfies(cond, x_prime, cy, circles):
            k = i
            break
    if k is None:
        k = len(conds) - 1  # fallback

    base = conds[k]
    print("base", base)
    base_in = set(base.get("in", set()))
    base_out = set(base.get("out", set()))
    base_ineq = list(base.get("ineq", []))

    # Tres regiones: down / middle / up
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

    # 1) Actualizar condiciones en sweepL (para conocer índices reales)
    new_conds = conds[:k] + [down, middle, up] + conds[k + 1 :]
    sweepL["conditions"] = new_conds
    sweepL.setdefault("active", set()).add(circle_id)

    # Índices de las bandas recién insertadas
    kb_down = k  # índice de 'down'
    kb_mid = k + 1  # índice de 'middle' (por si lo quieres loguear)
    kb_up = k + 2  # índice de 'up'

    # 2) Encolar intersecciones con círculos vecinos (arriba/abajo)
    ids = {circle_id}
    circles_nearby = collect_circles_from_neighbors(
        ids, kb_down, kb_up, sweepL["conditions"]
    )
    print("check intersections in circles", circles_nearby)
    Q = enqueue_intersections_from_ids(
        Q, circles, circles_nearby, sweepL=sweepL, eps=1e-9, keep_sorted=True, tol=1e-9
    )

    # 3) Agregar el right endpoint de este círculo y reordenar Q
    Q.append((cx + r, cy, "right", {circle_id}))
    order_Q(Q, circles)
    print_Q(Q, label="Q")
    print_sweepL(sweepL, label="L")

    return Q, sweepL


def mark_intersection_needed(cond, var="y"):
    """
    Return 'check for intersection' if `cond["ineq"]` contains BOTH '<' and '>'
    inequalities for the given variable (default 'y'). Supports ANY number of inequalities.
    Otherwise return None.
    """
    lt = gt = False
    for item in cond.get("ineq", []):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        v, op, _ = item
        if v != var:
            continue
        if op == "<":
            lt = True
        elif op == ">":
            gt = True
        if lt and gt:
            return True
    return None


def circle_circle_intersections(c1, c2, tol=1e-12):
    (x0, y0), r0 = c1
    (x1, y1), r1 = c2
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    # no solutions or infinite solutions (coincident) -> return none
    if d > r0 + r1 + tol:  # separate
        return []
    if d < abs(r0 - r1) - tol:  # contained
        return []
    if d < tol and abs(r0 - r1) < tol:  # coincident
        return []

    # base point along the line of centers
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h2 = r0 * r0 - a * a
    if h2 < -tol:
        return []
    h = math.sqrt(max(h2, 0.0))

    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    if h <= tol:  # tangent (one point)
        return [(xm, ym)]

    # two intersection points
    rx = -dy * (h / d)
    ry = dx * (h / d)
    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def intersections_from_condition(cond, circles, var="y"):
    """
    Collect circle IDs from cond['ineq'] (for `var`), compute pairwise intersections,
    and return Q-style tuples: (x, y, "intersection", {cid1, cid2}).
    """
    circle_ids = sorted({cid for v, _, (cid, _) in cond.get("ineq", []) if v == var})
    if len(circle_ids) < 2:
        return []

    events = []
    for i, j in combinations(circle_ids, 2):
        pts = circle_circle_intersections(circles[i - 1], circles[j - 1])
        for x, y in pts:
            events.append((x, y, "intersection", {i, j}))
    return events


# # Example:
# up = {'in': {11}, 'out': {11, 5}, 'ineq': [('y','<',(5,'y_low')), ('y','>',(11,'y_high'))]}
# print(intersections_from_condition(up, circles))


def rightend_point(Q, sweepL, circle_id, circles, eps=1e-6):
    """
    Procesa el extremo derecho del círculo `circle_id` en la posición sweepL['x'].
    - Busca la región 'middle' (condición con circle_id en 'in') que contiene (x - eps, cy)
    - Toma sus vecinas 'down' (k-1) y 'up' (k+1)
    - Opcional: detecta intersecciones en up/down (igual que en leftend_point)
    - Elimina la región 'middle' y fusiona up y down removiendo referencias al círculo
    - No agrega evento 'right' a Q
    """
    (cx, cy), r = circles[circle_id - 1]
    print("Círculo derecho con centro en (", cx, ",", cy, ") y radio ", r)

    x_prime = sweepL["x"] - eps  # evaluar justo antes del extremo derecho
    conds = sweepL["conditions"]
    print_sweepL(sweepL, label="Conditions")

    # 1) localizar la región 'middle' (con circle_id en 'in') que contiene (x', cy)
    k = None
    for i, cond in enumerate(conds):
        if circle_id in cond.get("in", set()) and satisfies(cond, x_prime, cy, circles):
            k = i
            break

    if k is None:
        # No se encontró región media; desactivar y salir
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

    down = conds[k - 1]
    mid = conds[k]
    up = conds[k + 1]

    print("down", down)
    print("middle", mid)
    print("up", up)

    # 3) Fusionar up y down en una región base, quitando referencias a circle_id
    def _sin_ineq_del_circulo(ineqs, cid):
        # elimina desigualdades que referencien a este círculo
        filtradas = []
        for t in ineqs:
            if not (isinstance(t, (list, tuple)) and len(t) == 3):
                filtradas.append(t)
                continue
            var, op, ref = t
            if var == "y" and isinstance(ref, tuple) and ref[0] == cid:
                continue  # quitarla
            filtradas.append(t)
        return filtradas

    # in/out base: intersección de ambos vecinos (y quitando el círculo de 'out')
    base_in = set(down.get("in", set())) & set(up.get("in", set()))
    base_out = (set(down.get("out", set())) & set(up.get("out", set()))) - {circle_id}

    # ineq base: unión de ineqs de up y down sin referencias al círculo que cierra (deduplicada)
    ineq_down = _sin_ineq_del_circulo(down.get("ineq", []), circle_id)
    ineq_up = _sin_ineq_del_circulo(up.get("ineq", []), circle_id)

    base_ineq = []
    for t in ineq_down + ineq_up:
        if t not in base_ineq:
            base_ineq.append(t)

    merged = {"in": base_in, "out": base_out, "ineq": base_ineq}
    print("merged", merged)

    # 4) Reemplazar [down, middle, up] por [merged]
    sweepL["conditions"] = conds[: k - 1] + [merged] + conds[k + 2 :]
    sweepL.setdefault("active", set()).discard(circle_id)

    sweepL = remove_circle_from_L(
        sweepL,
        circle_id,
        circles=circles,
        rebuild_triples=True,  # <<< rebuild full vertical partition at x
        keep_empty_base=True,
    )

    # print("L (tras cerrar círculo)", sweepL)
    print_sweepL(sweepL, label="L (tras cerrar círculo)")
    # Nota: NO agregamos evento 'right' a Q
    return Q, sweepL


# def input_point(sweepL, Ac, circles, q):
#     """
#     Remove from Ac any circle whose ID is in sweepL['active'] and that contains q.
#     IDs in sweepL['active'] are 1-based indexes into `circles`.
#     q can be (x, y) or (x, y, 'input', {...}).
#     """
#     if not (isinstance(q, tuple) and len(q) >= 2):
#         raise ValueError("q must be a tuple like (x,y) or (x,y,'input',{...})")
#     x, y = q[0], q[1]

#     active_ids = set(sweepL.get("active", set()))
#     if not active_ids:
#         return Ac

#     EPS = 1e-9  # for boundary-inclusive check
#     TOL = 1e-9  # to match circles in Ac by value

#     def point_in_circle(px, py, circle):
#         (cx, cy), r = circle
#         return (px - cx) ** 2 + (py - cy) ** 2 <= r**2 + EPS

#     def same_circle(a, b):
#         (ax, ay), ar = a
#         (bx, by), br = b
#         return abs(ax - bx) <= TOL and abs(ay - by) <= TOL and abs(ar - br) <= TOL

#     to_remove = []
#     for cid in active_ids:
#         if 1 <= cid <= len(circles):
#             c = circles[cid - 1]  # 1-based IDs
#             if point_in_circle(x, y, c):
#                 to_remove.append(c)

#     if not to_remove:
#         return Ac

#     return [c for c in Ac if not any(same_circle(c, r) for r in to_remove)]


def input_point(sweepL, Ac, circles, q, eps=1e-9):
    """
    Register the INPUT point into the matrix Ac using ONLY circles currently on the sweep line.
    - If the point is inside exactly one active circle i -> Ac[i,i] += 1
    - If inside k>=2 active circles -> increment all unordered pairs (i,j) symmetrically.
    Returns Ac (so your existing `Ac = input_point(...)` line still works).
    """
    # q is (x, y, ...) — we only need x,y
    if not (isinstance(q, tuple) and len(q) >= 2):
        return Ac
    x, y = q[0], q[1]

    # De-duplicate same input point (optional; avoids double counting if Q repeats it)
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
        return (px - cx) ** 2 + (py - cy) ** 2 <= r**2 + eps  # boundary-inclusive

    # Find which ACTIVE circles contain the point
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
        # increment all unordered pairs symmetrically
        for a in range(len(containing)):
            for b in range(a + 1, len(containing)):
                i, j = containing[a] - 1, containing[b] - 1
                Ac[i][j] += 1
                Ac[j][i] += 1
    return Ac


def register_point_membership(Ac, member_ids):
    S = sorted(member_ids)
    if not S:
        return Ac
    if len(S) == 1:
        Ac[S[0] - 1][S[0] - 1] += 1
        return Ac
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            i, j = S[a] - 1, S[b] - 1
            lo, hi = (i, j) if i < j else (j, i)
            Ac[lo][hi] += 1
    return Ac


def intersection_point(Q, sweepL, circles, ids, x, y, eps=1e-6):
    """
    Handle an intersection event at (x, y) between the two circle IDs in `ids`.
    Uses pick_conditions_for_ids + find_condition_index_at_from_subset to locate
    the down/middle/up bands just before and just after the intersection.
    """
    assert (
        isinstance(ids, (set, frozenset)) and len(ids) == 2
    ), "ids must be a set of two circle IDs"
    # a_id, b_id = sorted(ids)

    conds = sweepL["conditions"]

    # --- build the subset of conditions relevant to these ids ---
    conds_sel, idx_sel = pick_conditions_for_ids(ids, sweepL)  # your helper
    # subset = (conds_sel, idx_sel)
    subset = set(ids)

    # Tiny vertical probe to pick bands around y
    dy = max(1e-8, eps)

    # # BEFORE (x - eps)
    # x_before = x - eps
    # x_after = x + eps  # you already have this earlier

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

    # print("REGIONES ANTES")

    # print("*********DOWN")
    # kb_down   = find_condition_index_at_from_subset(x_before, y - dy, circles, conds, subset)
    # print("before down",kb_down, conds[kb_down])

    # print("*******UP")
    # kb_up     = find_condition_index_at_from_subset(x_before, y + dy,  circles, conds, subset)
    # print("before up",kb_up, conds[kb_up])

    print("************MIDDLE")
    kb_middle = pick_middle_index_between(conds, kb_down, kb_up, ids)
    if kb_middle is None:
        # fallback: try the second upward probe, or the exact y
        kb_middle = find_condition_index_at_from_subset_safe(
            x_before, y_up2, circles, conds, subset
        )
        if kb_middle is None:
            kb_middle = find_condition_index_at_from_subset_safe(
                x_before, y, circles, conds, subset
            )
    print("before middle", kb_middle, conds[kb_middle])

    # print("************MIDDLE")
    # kb_middle = pick_middle_index_between(conds, kb_down, kb_up, ids)
    # if kb_middle is None:
    #     # last-resort fallback if something degenerate happens
    #     kb_middle = find_condition_index_at_from_subset(x_before, y, circles, conds, subset)
    # print("before middle",kb_middle, conds[kb_middle])

    print("REGIONES DESPUES")
    # por definición
    ka_down = kb_down
    ka_up = kb_up

    # # middle-after = middle-before with ids purged, then re-evaluated at (x+eps, y)
    # middle_after = rebuild_middle_after_condition_general(conds, kb_middle, ids, x_after, y, circles, eps)

    # # Write it back in place; the middle band keeps the same slot after the crossing.
    # conds[kb_middle] = middle_after
    # ka_middle = kb_middle  # index now corresponds to the rebuilt middle-after band
    # print(middle_after)
    # print_sweepL(sweepL)

    # # --- build 3 middle-after candidates at different heights ---
    # cand_at, cand_up, cand_down = rebuild_middle_after_condition_by_sampling(
    #     conds, kb_middle, ids, x_after, y, circles, eps=eps, epsy=max(10*eps, 1e-7)
    # )
    # print("cand_at ", cand_at)
    # print("cand_up ", cand_up)
    # print("cand_down ", cand_down)

    # middle_after = choose_middle_candidate(
    # cand_at, cand_up, cand_down,
    # conds[kb_down], conds[kb_up]
    # )
    # # compare candidates with the actual up/down BEFORE regions (by in/out signature)
    # sig_up_before   = _region_signature(conds[kb_up])
    # sig_down_before = _region_signature(conds[kb_down])

    # candidates = [("at",   cand_at),
    #               ("up",   cand_up),
    #               ("down", cand_down)]

    # # choose the candidate whose signature is different from both up & down
    # chosen = None
    # for tag, cand in candidates:
    #     sig = _region_signature(cand)
    #     if sig != sig_up_before and sig != sig_down_before:
    #         chosen = cand
    #         break
    # if chosen is None:
    #     # fallback: prefer the "up" probe; if still equal, take "at"
    #     chosen = cand_up if _region_signature(cand_up) != sig_up_before else cand_at

    # # write back the middle-after
    # conds[kb_middle] = chosen
    # print(chosen)

    middle_after = rebuild_middle_after_by_flip(
        conds, kb_middle, ids, x_after, y, circles, eps
    )

    conds[kb_middle] = middle_after
    print(middle_after)
    print_sweepL(sweepL)

    # Intersecciones
    circles_nearby = collect_circles_from_neighbors(
        ids, kb_down, kb_up, sweepL["conditions"]
    )
    print("check intersections in circles", circles_nearby)
    Q = enqueue_intersections_from_ids(
        Q, circles, circles_nearby, sweepL=sweepL, eps=1e-9, keep_sorted=True, tol=1e-9
    )

    order_Q(Q, circles)

    # Done
    print_sweepL(sweepL, label="L (post-intersection)")
    print_Q(Q, label="Q")
    return Q, sweepL


def rebuild_middle_after_by_flip(conds, kb_middle, ids, x_after, y, circles, eps=1e-9):
    """
    Rebuild middle-after by *flipping* membership of the two intersecting circles in `ids`.
    Keeps all non-ids as in the original middle. For ids that become 'out', add a symbolic
    bound ('< y_low' or '> y_high') decided by whether y is closer to the lower or upper band.
    """
    base = conds[kb_middle]

    # purge prior references to ids
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
            # now becomes 'out' – choose lower or upper inequality symbolically
            (cx, cy), r = circles[cid - 1]
            yl, yh = _circle_y_band_at_x(
                circles[cid - 1], x_after
            )  # your existing helper

            # Robust choice that avoids vertex-equality issues: compare to the mid-height
            if yl is None:
                # circle not vertically present at x_after: just mark out (no inequality)
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
            # was out → becomes in
            outc["in"].add(cid)

    return outc


def pick_conditions_for_ids(ids, sweepL):
    """
    ids: {a, b} con exactamente dos IDs de círculos (enteros 1-based).
    sweepL: dict con clave 'conditions' (lista de bandas ordenadas verticalmente).

    Devuelve (conds_sel, idx_sel):
      - conds_sel: lista de dicts de condiciones seleccionadas en orden vertical.
      - idx_sel:   lista de índices correspondientes en sweepL['conditions'].
    """
    assert (
        isinstance(ids, (set, frozenset)) and len(ids) == 2
    ), "ids debe tener exactamente dos elementos"
    a, b = sorted(ids)
    conds = sweepL.get("conditions", [])

    # 1) Índices donde 'in' contiene ambos IDs
    both_idxs = [i for i, c in enumerate(conds) if ids.issubset(c.get("in", set()))]

    if both_idxs:
        # Tomar el bloque contiguo mínimo que cubre todos los 'both'
        start = min(both_idxs)
        end = max(both_idxs)

        # Expandir con dos arriba y dos abajo si existen
        lo = max(0, start - 2)
        hi = min(len(conds) - 1, end + 2)
        idx_sel = list(range(lo, hi + 1))
        conds_sel = [conds[i] for i in idx_sel]
        return conds_sel, idx_sel

    # 2) No hay ninguna banda con ambos; buscar la pareja más cercana a y b
    idxs_a = [i for i, c in enumerate(conds) if a in c.get("in", set())]
    idxs_b = [i for i, c in enumerate(conds) if b in c.get("in", set())]

    # Fallback si alguno no aparece: devolver una ventanita centrada en el que sí aparece
    if not idxs_a and not idxs_b:
        return [], []
    if not idxs_a or not idxs_b:
        base = (idxs_a or idxs_b)[0]
        lo = max(0, base - 1)
        hi = min(len(conds) - 1, base + 1)
        idx_sel = list(range(lo, hi + 1))
        conds_sel = [conds[i] for i in idx_sel]
        return conds_sel, idx_sel

    # Elegir el par (i, j) con distancia mínima |i - j|
    best = None
    for i in idxs_a:
        for j in idxs_b:
            dist = abs(i - j)
            if best is None or dist < best[0]:
                best = (dist, i, j)

    _, i, j = best
    low, high = (i, j) if i < j else (j, i)

    # Idealmente queremos una banda "entre" (low+1). Si no la hay, ajustamos.
    if high - low >= 2:
        middle = low + 1
        idx_sel = [low, middle, high]
    else:
        # No hay banda estrictamente en medio; intentamos construir 3 tomando vecinos si existen
        idx_sel = [low, high]
        # Añadir un vecino razonable para tener 3 si es posible
        neighbor = None
        if low - 1 >= 0:
            neighbor = low - 1
        elif high + 1 < len(conds):
            neighbor = high + 1
        if neighbor is not None:
            # Ordenar verticalmente
            idx_sel = sorted(set(idx_sel + [neighbor]))

    conds_sel = [conds[k] for k in idx_sel]
    return conds_sel, idx_sel


def find_condition_index_at_from_subset_safe(x, y, circles, conds, subset, tol=1e-12):
    # 1) exact match using only the subset
    ins_sub, out_sub = _classify_inside_outside(x, y, circles, subset, tol)

    candidates = []
    for i, c in enumerate(conds):
        # subset must be consistent with the band's fixed memberships
        if not (c["in"] & subset <= ins_sub):  # band requires inside but we're not
            continue
        if not (c["out"] & subset <= out_sub):  # band requires outside but we're not
            continue
        # all inequalities present in the band must hold at (x,y)
        ok = True
        for _var, op, (cid, which) in c.get("ineq", []):
            if not _ineq_holds(y, op, cid, which, x, circles, tol):
                ok = False
                break
        if ok:
            candidates.append(i)

    if candidates:
        # bands are vertically ordered in 'conds'; take the first match
        return candidates[0]

    # 2) fallback: use *all* ids mentioned anywhere in the conditions, then score
    all_ids = set()
    for c in conds:
        all_ids |= (
            c["in"] | c["out"] | {cid for (_v, _op, (cid, _w)) in c.get("ineq", [])}
        )

    ins_all, out_all = _classify_inside_outside(x, y, circles, all_ids, tol)

    best_i, best_score = None, float("inf")
    for i, c in enumerate(conds):
        score = 0
        # penalties for in/out mismatches (only for ids this band cares about)
        score += len(c["in"] - ins_all)
        score += len(c["out"] - out_all)
        # penalties for ineq violations, weighted by boundary distance
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

import math


def _circle_y_band(circles, cid, x, tol=1e-15):
    """Return (y_low, y_high) for circle cid at x; (None, None) if no intersection."""
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
    """Check an inequality like ('y','<',(cid,'y_low'|'y_high')) at (x,y)."""
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
    return False  # unknown operator


def _classify_inside_outside(x, y, circles, ids, tol=1e-12):
    """Return (inside_set, outside_set) for the ids at (x,y)."""
    inside, outside = set(), set()
    for cid in ids:
        (cx, cy), r = circles[cid - 1]
        val = (x - cx) * (x - cx) + (y - cy) * (y - cy) - r * r
        (inside if val <= tol else outside).add(cid)
    return inside, outside


import math
from copy import deepcopy


def _circle_y_band_at_x(circle, x, tol=1e-12):
    """Return (y_low, y_high) of the circle at vertical line x, or (None, None) if |x-cx|>r."""
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
    """Remove any mentions of the circles in `ids` from in/out/ineq."""
    newc = {"in": set(), "out": set(), "ineq": []}
    newc["in"] = {c for c in cond.get("in", set()) if c not in ids}
    newc["out"] = {c for c in cond.get("out", set()) if c not in ids}
    for item in cond.get("ineq", []):
        # item is like ('y','<', (cid,'y_low')) or ('y','>', (cid,'y_high'))
        if (
            isinstance(item, (list, tuple))
            and len(item) == 3
            and isinstance(item[2], (list, tuple))
            and len(item[2]) == 2
        ):
            cid = item[2][0]
            if cid in ids:
                continue  # drop it
        newc["ineq"].append(item)
    return newc


# def rebuild_middle_after_condition(conds, kb_middle, ids, x_after, y, circles, eps=1e-9):
#     """
#     Take the *before* middle condition at index `kb_middle`, remove any references to `ids`,
#     then re-evaluate for each cid in `ids` whether at (x_after, y) the band should be:
#       - inside that circle (-> add to 'in')
#       - above it      (-> add to 'out' and ('y','>',(cid,'y_high')))
#       - below it      (-> add to 'out' and ('y','<', (cid,'y_low')))
#     Returns the rebuilt condition (does not modify `conds` in place).
#     """
#     base = conds[kb_middle]
#     outc = _purge_ids_from_condition(base, ids)

#     # Re-evaluate for each circle in ids at (x_after, y)
#     for cid in sorted(ids):
#         yl, yh = _circle_y_band_at_x(circles[cid - 1], x_after)
#         if yl is None:   # vertical line does not cut circle (shouldn't happen near intersection, but safe-guard)
#             # Treat as outside without adding an inequality (no band at this x)
#             outc["out"].add(cid)
#             continue

#         if y < yl - eps:
#             outc["out"].add(cid)
#             outc["ineq"].append(("y", "<", (cid, "y_low")))
#         elif y > yh + eps:
#             outc["out"].add(cid)
#             outc["ineq"].append(("y", ">", (cid, "y_high")))
#         else:
#             outc["in"].add(cid)

#     return outc


def _in_set(c):
    return set(c.get("in", set()))


def choose_middle_candidate(cand_at, cand_up, cand_down, down_before, up_before):
    in_down = _in_set(down_before)
    in_up = _in_set(up_before)
    k_down, k_up = len(in_down), len(in_up)

    candidates = [cand_at, cand_up, cand_down]

    if k_up == k_down:
        # tangency/degenerate—stick with the “at” probe
        return cand_at

    if k_up > k_down:
        # lower vertex: middle moves toward the region with MORE circles inside
        return max(candidates, key=lambda c: len(_in_set(c)))
    else:
        # upper vertex: middle moves toward the region with FEWER circles inside
        return min(candidates, key=lambda c: len(_in_set(c)))

import math


def _classify_middle_before(cond, ids):
    """Return 'between', 'inside_both', or 'mixed' for the middle-before band."""
    ids = set(ids)
    c_in = set(cond.get("in", set()))
    c_out = set(cond.get("out", set()))
    if ids.issubset(c_in):
        return "inside_both"
    if ids.isdisjoint(c_in) and ids.issubset(c_out):
        # BETWEEN has both '<' and '>' involving the two ids
        lt = gt = False
        for item in cond.get("ineq", []):
            if (not isinstance(item, (list, tuple))) or len(item) != 3:
                continue
            v, op, rhs = item
            if v != "y" or not isinstance(rhs, tuple):
                continue
            cid, _ = rhs
            if cid in ids:
                if op == "<":
                    lt = True
                elif op == ">":
                    gt = True
        if lt and gt:
            return "between"
    return "mixed"


# def rebuild_middle_after_condition_general(conds, kb_middle, ids, x_after, y, circles, eps=1e-9):
#     """
#     Robust rebuild of the middle-after band. Decides using the *type of the middle-before*
#     and toggles BETWEEN <-> INSIDE_BOTH. Falls back to probe logic if ambiguous.
#     """
#     before = conds[kb_middle]
#     mode_before = _classify_middle_before(before, ids)

#     # purge old references to `ids`
#     outc = _purge_ids_from_condition(before, ids)

#     def y_band(circle, x):
#         (cx, cy), r = circle
#         dx = x - cx
#         R2 = r*r - dx*dx
#         if R2 < 0:
#             return (None, None)
#         h = math.sqrt(max(R2, 0.0))
#         return (cy - h, cy + h)

#     # ---- canonical toggles ----
#     if mode_before == 'between':
#         # AFTER must be inside both
#         outc["in"].update(ids)
#         return outc

#     if mode_before == 'inside_both':
#         # AFTER must be between the two arcs.
#         (a, b) = sorted(ids)
#         yl_a, yh_a = y_band(circles[a-1], x_after)
#         yl_b, yh_b = y_band(circles[b-1], x_after)

#         # choose lower by smaller y_high; upper by larger y_low (guarding None)
#         pairs = [(a, yl_a, yh_a), (b, yl_b, yh_b)]
#         lower_cid = min(pairs, key=lambda t: (float('inf') if t[2] is None else t[2]))[0]
#         upper_cid = max(pairs, key=lambda t: (-float('inf') if t[1] is None else t[1]))[0]

#         outc["out"].update(ids)
#         outc["ineq"].append(("y", ">", (lower_cid, "y_high")))
#         outc["ineq"].append(("y", "<", (upper_cid, "y_low")))
#         return outc

#     # ---- fallback: probe logic (handles odd tangencies etc.) ----
#     def classify_at_y(circle, x, y0, tol):
#         yl, yh = y_band(circle, x)
#         if yl is None:
#             return "outside"
#         if y0 < yl - tol:  return "below"
#         if y0 > yh + tol:  return "above"
#         if abs(y0 - yl) <= tol: return "edge_low"
#         if abs(y0 - yh) <= tol: return "edge_high"
#         return "inside"

#     d = max(10*eps, 1e-7)
#     y_minus, y_plus = y - d, y + d
#     inside_minus, inside_plus = set(), set()

#     for cid in sorted(ids):
#         c = circles[cid-1]
#         if classify_at_y(c, x_after, y_minus, eps) == "inside":
#             inside_minus.add(cid)
#         if classify_at_y(c, x_after, y_plus,  eps) == "inside":
#             inside_plus.add(cid)

#     if inside_minus == ids and inside_plus == ids:
#         outc["in"].update(ids)
#         return outc

#     if len(inside_minus) == 1 and len(inside_plus) == 1 \
#        and inside_minus != inside_plus and inside_minus | inside_plus == ids:
#         # BETWEEN
#         lower_cid = next(iter(inside_minus))
#         upper_cid = next(iter(inside_plus))
#         outc["out"].update(ids)
#         outc["ineq"].append(("y", ">", (lower_cid, "y_high")))
#         outc["ineq"].append(("y", "<", (upper_cid, "y_low")))
#         return outc

#     if len(inside_minus) == 1 and inside_minus == inside_plus:
#         cid_in  = next(iter(inside_minus))
#         cid_out = next(iter(ids - {cid_in}))
#         outc["in"].add(cid_in)
#         yl, yh = y_band(circles[cid_out-1], x_after)
#         if yl is None:
#             outc["out"].add(cid_out)
#         elif y < yl - eps:
#             outc["out"].add(cid_out); outc["ineq"].append(("y","<",(cid_out,"y_low")))
#         elif y > yh + eps:
#             outc["out"].add(cid_out); outc["ineq"].append(("y",">",(cid_out,"y_high")))
#         else:
#             outc["in"].add(cid_out)
#         return outc

#     # Last-resort: keep both inside (most stable default)
#     outc["in"].update(ids)
#     return outc

import math


def _classify_at_probe(base_cond, ids, x_after, y_probe, circles, eps=1e-9):
    """
    Purge ids from base_cond, then classify each cid in ids at (x_after, y_probe)
    and rebuild the condition.
    """
    outc = _purge_ids_from_condition(base_cond, ids)  # keeps other circles' constraints
    for cid in sorted(ids):
        y_low, y_high = _circle_y_band_at_x(circles[cid - 1], x_after)
        if y_low is None:
            outc["out"].add(cid)
            continue
        if y_probe < y_low - eps:
            outc["out"].add(cid)
            outc["ineq"].append(("y", "<", (cid, "y_low")))
        elif y_probe > y_high + eps:
            outc["out"].add(cid)
            outc["ineq"].append(("y", ">", (cid, "y_high")))
        else:
            outc["in"].add(cid)
    return outc


def _region_signature(region):
    """Compare regions by (in,out) sets only (ignore inequality ordering)."""
    return (frozenset(region.get("in", set())), frozenset(region.get("out", set())))


def rebuild_middle_after_condition_by_sampling(
    conds, kb_middle, ids, x_after, y, circles, eps=1e-9, epsy=None
):
    """
    Build three candidates at (x_after, y), (x_after, y+epsy), (x_after, y-epsy),
    then return the one whose (in,out) is different from BOTH the up and down bands.
    If all collide, return the 'y+epsy' candidate as a tie-breaker.
    """
    if epsy is None:
        epsy = max(10 * eps, 1e-7)

    base = conds[kb_middle]

    # make fresh base snapshots using literal dicts (not dict(...))
    base_at = {
        "in": set(base.get("in", set())),
        "out": set(base.get("out", set())),
        "ineq": list(base.get("ineq", [])),
    }
    base_up = {
        "in": set(base.get("in", set())),
        "out": set(base.get("out", set())),
        "ineq": list(base.get("ineq", [])),
    }
    base_down = {
        "in": set(base.get("in", set())),
        "out": set(base.get("out", set())),
        "ineq": list(base.get("ineq", [])),
    }

    cand_at = _classify_at_probe(base_at, ids, x_after, y, circles, eps)
    cand_up = _classify_at_probe(base_up, ids, x_after, y + epsy, circles, eps)
    cand_down = _classify_at_probe(base_down, ids, x_after, y - epsy, circles, eps)

    return cand_at, cand_up, cand_down


def _circle_ids_from_condition(cond):
    """Extract all circle IDs mentioned in a condition (in, out, and ineq)."""
    s = set()
    # in / out
    s |= set(cond.get("in", set()))
    s |= set(cond.get("out", set()))
    # inequalities: ('y','<', (cid,'y_low')) or ('y','>', (cid,'y_high'))
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
    From conds[kb_down] and its previous condition, and conds[kb_up] and its next condition,
    collect every circle ID referenced (in/out/ineq). Add the given `ids` as well.
    Return a *list* of unique circle IDs (order by first appearance in this scan).
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

    # kb_down and neighbor before
    add_from_index(kb_down)
    if isinstance(kb_down, int):
        add_from_index(kb_down - 1)

    # kb_up and neighbor after
    add_from_index(kb_up)
    if isinstance(kb_up, int):
        add_from_index(kb_up + 1)

    # include the intersecting pair `ids`
    for cid in ids:
        if cid not in order:
            order.append(cid)

    return order


def intersections_from_ids(ids_list, circles):
    """
    Given a list of circle IDs (1-based), compute all pairwise intersections
    and return Q-style events: (x, y, 'intersection', {i, j}).
    """
    # unique, valid, preserve order
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
    Build intersection events from ids_list and add them to Q:
      - only keep events strictly ahead of the sweep line (x > sweepL['x'] + eps) if sweepL provided
      - deduplicate against existing Q within `tol`
      - keep Q sorted (uses order_Q if available)
    Returns the updated Q.
    """
    new_events = intersections_from_ids(ids_list, circles)

    # keep only future events if sweepL is provided
    if sweepL is not None and "x" in sweepL:
        cutoff = sweepL["x"] + eps
        new_events = [ev for ev in new_events if ev[0] > cutoff]

    # If you already have add_unique_events, use it:
    try:
        add_unique_events(Q, new_events, tol=tol, keep_sorted=keep_sorted)
    except NameError:
        # Fallback: simple dedup + append
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

        # try to sort if order_Q exists
        try:
            order_Q(Q, circles)
        except NameError:
            Q.sort(key=lambda e: (e[0], e[1]))

    # Optional: ensure final sort
    try:
        order_Q(Q, circles)
    except NameError:
        Q.sort(key=lambda e: (e[0], e[1]))

    return Q


def _cond_class_vs_ids(cond, ids):
    """Classify a condition vs the two ids: 'inside_both', 'between',
    'one_inside', or 'other'."""
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
    """Pick the true middle index between kb_down and kb_up."""
    lo = min(kb_down, kb_up)
    hi = max(kb_down, kb_up)
    if hi - lo < 2:
        return None  # nothing in between

    best_one_inside = None
    for i in range(lo + 1, hi):
        typ = _cond_class_vs_ids(conds[i], ids)
        if typ in ("inside_both", "between"):
            return i
        if typ == "one_inside" and best_one_inside is None:
            best_one_inside = i

    # fallback: take the strongest we saw, or just the slot between
    if best_one_inside is not None:
        return best_one_inside
    return lo + 1


# def pick_middle_index_between(conds, i, j, ids=None):
#     if not conds:
#         return None
#     if i is None and j is None:
#         return min(len(conds)//2, len(conds)-1)
#     if i is None:
#         return j
#     if j is None:
#         return i
#     lo, hi = (i, j) if i <= j else (j, i)
#     mid = lo + max(1, (hi - lo)//2)
#     return max(0, min(mid, len(conds)-1))

import math


def _safe_circle_band_at_x(circle, x, tol=1e-15):
    """Like your _circle_y_band_at_x but defensive."""
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
    Compute an x-nudge and an upward-only dy that stays strictly inside
    the bands of all circles in `ids` at x-eps and x+eps.
    Returns: (x_before, x_after, dy, y_up, y_up2)
    """
    # horizontal nudge scales with problem size
    max_r = max(circles[cid - 1][1] for cid in ids)
    epsx = max(x_eps_min, x_eps_scale * (abs(x) + max_r))

    x_before = x - epsx
    x_after = x + epsx

    # collect bands and clearances at x_before
    bands = []
    clears = []
    for cid in sorted(ids):
        yl, yh = _safe_circle_band_at_x(circles[cid - 1], x_before)
        if yl is None:  # degenerate (shouldn't happen very near an intersection),
            continue  # but just skip in the min()s below
        bands.append(yh - yl)
        # distance from y to each boundary; clamp negative to 0
        c_low = max(0.0, y - yl)
        c_high = max(0.0, yh - y)
        clears.append(min(c_low, c_high))

    # safe defaults if something degenerate slips through
    if not bands:
        bands = [1.0]
    if not clears:
        clears = [1.0]

    # choose dy small enough to (a) be well above numeric noise, (b) stay *inside* every band
    dy_raw = frac_band * min(bands)  # 10% of thinnest band
    dy_clear = frac_clear * min(clears)  # keep well away from boundaries
    dy_floor = floor_mult * epsx  # above numerical floor
    dy = max(dy_floor, min(dy_raw, dy_clear))

    # two upward probes; clamp the second to remain inside the closest clearance
    y_up = y + dy
    y_up2 = y + min(2.0 * dy, 0.9 * min(clears))  # 0.9 keeps a buffer from the arc

    return x_before, x_after, dy, y_up, y_up2


# def algoritmo(circles, points):
#     # Inicializar variables
#     Ac = list(circles)  # copia independiente
#     Q = build_Q(circles, points)
#     eps = 1e-8
#     x0 = Q[0][0] - eps
#     sweepL = {
#         "x": x0,
#         "active": set(),
#         "conditions": [{"in": set(), "out": set(), "ineq": []}],
#     }

#     print("Antes de iniciar:")
#     print_Q(Q, label="Q")
#     print("Línea de barrido =", sweepL)
#     print("Ac =", Ac)

#     # helper para mapear círculos en Ac a IDs (1-based) respecto a 'circles'
#     def ac_ids(ac_list):
#         ids = set()
#         for i, c in enumerate(circles, start=1):
#             if c in ac_list:
#                 ids.add(i)
#         return ids

#     i = 0
#     while i < len(Q):
#         print_Q(Q, label="Q")
#         x, y, kind, idx = Q[i]

#         advance_sweep_to(sweepL, x, circles)

#         if kind == "input":
#             print(
#                 f"#####     Point ({x}, {y}) is an INPUT point, index {idx}     #####"
#             )
#             before_ids = ac_ids(Ac)
#             new_Ac = input_point(sweepL, Ac, circles, Q[i])
#             after_ids = ac_ids(new_Ac)
#             removed = sorted(before_ids - after_ids)
#             if removed:
#                 print(f" -> Removed from Ac (by ID): {removed}")
#             else:
#                 print(" -> No active circle contained this input point.")
#             Ac = new_Ac

#         elif kind == "left":
#             circle_idx = list(idx)[0]
#             print(
#                 f"#####     Point ({x}, {y}) is a LEFT endpoint of circle {idx}     #####"
#             )
#             Q, sweepL = leftend_point(
#                 Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6
#             )

#         elif kind == "right":
#             circle_idx = list(idx)[0]
#             print(
#                 f"#####     Point ({x}, {y}) is a RIGHT endpoint of circle {idx}     #####"
#             )
#             Q, sweepL = rightend_point(
#                 Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6
#             )

#         elif kind == "intersection":
#             # NEW: call your intersection handler
#             ids_set = set(idx)  # idx is already a set like {i, j}
#             print(
#                 f"#####     Point ({x}, {y}) is a INTERSECTION point of circles {ids_set}     #####"
#             )
#             Q, sweepL = intersection_point(Q, sweepL, circles, ids_set, x, y, eps=1e-6)

#         i += 1

#     print("Ac: ", Ac)
#     return Ac, Q, sweepL


def algoritmo(circles, points):
    n = len(circles)
    Ac = [[0]*n for _ in range(n)] # matrix of counters
    Q = build_Q(circles, points)
    eps = 1e-8
    x0 = Q[0][0] - eps
    sweepL = {
        "x": x0,
        "active": set(),
        "conditions": [{"in": set(), "out": set(), "ineq": []}
                      ],
        "seen_inputs": set(),  # to avoid double counting the same input
    }

    print("Antes de iniciar:")
    print_Q(Q, label="Q")
    print("Línea de barrido =", sweepL)
    print("Ac (matriz):")
    for row in Ac: print(row)

    i = 0
    while i < len(Q):
        print_Q(Q, label="Q")
        x, y, kind, idx = Q[i]
        advance_sweep_to(sweepL, x, circles)

        if kind == "input":
            print(f"#####     Point ({x}, {y}) is an INPUT point     #####")
            Ac = input_point(sweepL, Ac, circles, Q[i])  # <-- now registers in matrix

        elif kind == "left":
            circle_idx = list(idx)[0]
            print(f"#####     Point ({x}, {y}) is a LEFT endpoint of circle {idx}     #####")
            Q, sweepL = leftend_point(Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6)

        elif kind == "right":
            circle_idx = list(idx)[0]
            print(f"#####     Point ({x}, {y}) is a RIGHT endpoint of circle {idx}     #####")
            Q, sweepL = rightend_point(Q, sweepL, circle_id=circle_idx, circles=circles, eps=1e-6)

        elif kind == "intersection":
            ids_set = set(idx)
            print(f"#####     Point ({x}, {y}) is an INTERSECTION point of circles {ids_set}     #####")
            # (optional) also register intersections in the matrix
            Q, sweepL = intersection_point(Q, sweepL, circles, ids_set, x, y, eps=1e-6)

        i += 1

    print("Ac (final):")
    for row in Ac: print(row)
    return Ac, Q, sweepL

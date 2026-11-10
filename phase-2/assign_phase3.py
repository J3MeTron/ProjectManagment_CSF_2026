"""Фаза 2: распределение задач фазы 3 между участниками (задача о назначениях, Pyomo + HiGHS).

Данные: Оценки_задач_фазы3.xlsx (согласованная трудоемкость h_i и предпочтения p_ij).

Постановка (основная модель A):
  x_ij in {0, 1} - задача i назначена участнику j
  L_j = sum_i h_i x_ij + C            - нагрузка участника (C - общие активности, одинаковы у всех)
  max  sum_ij p_ij * h_i * x_ij       - суммарное "удовольствие", взвешенное часами:
                                        чем дольше задача, тем важнее отдать ее тому, кто ее хочет
  s.t. sum_j x_ij = 1                 - каждая задача ровно у одного исполнителя
       |L_j - L_avg| <= delta * L_avg - нагрузка отклоняется от средней не более чем на delta
       sum_i x_ij >= k_min            - у каждого не меньше k_min задач

Сравнительная модель B: min L_max (самая ровная нагрузка), предпочтения не учитываются.
Сравнительная модель C: max sum p_ij h_i x_ij без ограничения на баланс.

Запуск:  python assign_phase3.py [путь_к_xlsx] [--delta 0.15] [--kmin 3]
Зависимости: pyomo, highspy, openpyxl
"""
import argparse
from pathlib import Path
from statistics import median

import pyomo.environ as pyo
from openpyxl import load_workbook

R0 = 5
COMMON_HOURS = 4.5   # Ф3-14...Ф3-17: покер 2 + TCF/ECF 0.5 + два митинга по 1


def read_data(path):
    wb = load_workbook(path)
    ref = wb["Справочник"]
    tasks = []
    r = R0
    while ref.cell(r, 1).value:
        tasks.append((ref.cell(r, 1).value, ref.cell(r, 2).value))
        r += 1
    er = R0 + len(tasks) + 3
    experts = [ref.cell(er + e, 2).value for e in range(3)]
    h, p = {}, {}
    for i, _ in enumerate(tasks):
        finals = []
        for e in range(3):
            ws = wb[f"Эксперт{e + 1}"]
            r1, r2, pref = (ws.cell(R0 + i, col).value for col in (3, 4, 5))
            val = r2 if r2 not in (None, "") else r1
            if val in (None, "") or pref in (None, ""):
                raise ValueError(f"Эксперт{e + 1}: не заполнена строка {tasks[i][0]}")
            finals.append(float(val))
            p[i, e] = int(pref)
        h[i] = median(finals)
    return tasks, experts, h, p


def solve(tasks, experts, h, p, mode="A", delta=0.15, kmin=3):
    I, J = range(len(tasks)), range(len(experts))
    total = sum(h.values()) + COMMON_HOURS * len(experts)
    avg = total / len(experts)

    m = pyo.ConcreteModel()
    m.x = pyo.Var(I, J, domain=pyo.Binary)
    m.L = pyo.Expression(J, rule=lambda m, j: sum(h[i] * m.x[i, j] for i in I) + COMMON_HOURS)
    m.one = pyo.Constraint(I, rule=lambda m, i: sum(m.x[i, j] for j in J) == 1)
    m.kmin = pyo.Constraint(J, rule=lambda m, j: sum(m.x[i, j] for i in I) >= kmin)
    utility = sum(p[i, j] * h[i] * m.x[i, j] for i in I for j in J)

    if mode == "A":
        m.lo = pyo.Constraint(J, rule=lambda m, j: m.L[j] >= (1 - delta) * avg)
        m.hi = pyo.Constraint(J, rule=lambda m, j: m.L[j] <= (1 + delta) * avg)
        m.obj = pyo.Objective(expr=utility, sense=pyo.maximize)
    elif mode == "B":
        m.Lmax = pyo.Var(domain=pyo.NonNegativeReals)
        m.cap = pyo.Constraint(J, rule=lambda m, j: m.L[j] <= m.Lmax)
        m.obj = pyo.Objective(expr=m.Lmax, sense=pyo.minimize)
    else:
        m.obj = pyo.Objective(expr=utility, sense=pyo.maximize)

    res = pyo.SolverFactory("appsi_highs").solve(m)
    status = str(res.solver.termination_condition)
    assign = {i: next(j for j in J if pyo.value(m.x[i, j]) > 0.5) for i in I}
    loads = [pyo.value(m.L[j]) for j in J]
    util = sum(p[i, assign[i]] * h[i] for i in I)
    return status, assign, loads, util, avg


def report(name, tasks, experts, h, p, out):
    status, assign, loads, util, avg = out
    print(f"\n=== Модель {name} ({status}) ===")
    print(f"{'ID':<7}{'Задача':<48}{'ч':>5}  Исполнитель (предпочтение)")
    for i, (tid, t) in enumerate(tasks):
        j = assign[i]
        print(f"{tid:<7}{t[:46]:<48}{h[i]:>5.1f}  {experts[j]} ({p[i, j]})")
    for j, e in enumerate(experts):
        n = sum(1 for i in assign if assign[i] == j)
        mp = sum(p[i, j] for i in assign if assign[i] == j) / max(n, 1)
        print(f"  {e:<26} нагрузка {loads[j]:5.1f} ч, задач {n}, среднее предпочтение {mp:.2f}")
    print(f"  средняя нагрузка {avg:.1f} ч; разброс {max(loads) - min(loads):.1f} ч; "
          f"ЦФ sum p*h = {util:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="?", default=str(Path(__file__).parent / "Оценки_задач_фазы3.xlsx"))
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--kmin", type=int, default=3)
    a = ap.parse_args()
    data = read_data(a.xlsx)
    tasks, experts, h, p = data
    print(f"Задач: {len(tasks)}, суммарная трудоемкость {sum(h.values()):.1f} ч "
          f"+ общие активности {COMMON_HOURS} ч на человека")
    report(f"A: max предпочтений при балансе +-{a.delta:.0%}", *data,
           solve(*data, "A", a.delta, a.kmin))
    report("B: min максимальной нагрузки", *data, solve(*data, "B", a.delta, a.kmin))
    report("C: max предпочтений без баланса", *data, solve(*data, "C", a.delta, a.kmin))

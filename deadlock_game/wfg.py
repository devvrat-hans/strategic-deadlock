"""Wait-For Graph utilities (Definition 3)."""
from __future__ import annotations

from typing import Iterable


def cycle_members(n: int, edges: Iterable[tuple[int, int]]) -> frozenset[int]:
    """Return C(G): the set of nodes lying on some directed cycle.

    Works for arbitrary digraphs (Tarjan SCC); a node is on a cycle iff its SCC has
    more than one node or it has a self-loop.
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    self_loop = set()
    for i, j in edges:
        adj[i].append(j)
        if i == j:
            self_loop.add(i)
    index = 0
    idx = [-1] * n
    low = [0] * n
    on = [False] * n
    stack: list[int] = []
    members: set[int] = set()

    def strong(v: int) -> None:
        nonlocal index
        idx[v] = low[v] = index
        index += 1
        stack.append(v)
        on[v] = True
        for w in adj[v]:
            if idx[w] == -1:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on[w]:
                low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or v in self_loop:
                members.update(comp)

    for v in range(n):
        if idx[v] == -1:
            strong(v)
    return frozenset(members)

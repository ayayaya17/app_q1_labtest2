import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Fixed parameters (as required)
# -----------------------------
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
FITNESS_PEAK_ONES = 40  # fitness peaks at ones = 40
MAX_FITNESS = 80        # max fitness = 80 (when exactly 40 ones)

# -----------------------------
# GA helpers
# -----------------------------
def fitness_fn(bits: np.ndarray) -> int:
    """
    Peak fitness when number of ones == 40.
    Scale so that best possible fitness is 80.
    Simple symmetric penalty away from 40 ones.
    """
    ones = int(bits.sum())
    # Each step away from 40 reduces fitness by 2 (so 0..80 range stays nice)
    score = MAX_FITNESS - 2 * abs(ones - FITNESS_PEAK_ONES)
    return max(0, int(score))

def init_population(n: int, length: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(n, length), dtype=np.int8)

def tournament_select(pop: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
    """Pick best out of k random individuals."""
    idxs = np.random.randint(0, len(pop), size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return pop[best].copy()

def one_point_crossover(p1: np.ndarray, p2: np.ndarray, p_cross: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() > p_cross:
        return p1.copy(), p2.copy()
    point = random.randint(1, len(p1) - 2)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1.astype(np.int8), c2.astype(np.int8)

def mutate(bits: np.ndarray, p_mut: float = 0.01) -> np.ndarray:
    mask = np.random.rand(len(bits)) < p_mut
    bits[mask] = 1 - bits[mask]
    return bits

@dataclass
class GAResult:
    best_bits: np.ndarray
    best_fitness: int
    best_ones: int
    history: pd.DataFrame

def run_ga(
    pop_size: int,
    length: int,
    generations: int,
    elitism_n: int = 5,
    p_cross: float = 0.9,
    p_mut: float = 0.01,
) -> GAResult:
    pop = init_population(pop_size, length)

    hist = []
    best_bits_ever = None
    best_fit_ever = -1

    for gen in range(generations):
        fit = np.array([fitness_fn(ind) for ind in pop], dtype=np.int32)
        order = np.argsort(-fit)
        pop = pop[order]
        fit = fit[order]

        best_fit = int(fit[0])
        avg_fit = float(fit.mean())
        best_ones = int(pop[0].sum())

        if best_fit > best_fit_ever:
            best_fit_ever = best_fit
            best_bits_ever = pop[0].copy()

        hist.append({
            "generation": gen,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "best_ones": best_ones,
        })

        # Elitism
        new_pop = [pop[i].copy() for i in range(elitism_n)]

        # Create rest via selection + crossover + mutation
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fit, k=3)
            p2 = tournament_select(pop, fit, k=3)
            c1, c2 = one_point_crossover(p1, p2, p_cross=p_cross)
            c1 = mutate(c1, p_mut=p_mut)
            c2 = mutate(c2, p_mut=p_mut)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop, dtype=np.int8)

    best_bits = best_bits_ever
    return GAResult(
        best_bits=best_bits,
        best_fitness=int(fitness_fn(best_bits)),
        best_ones=int(best_bits.sum()),
        history=pd.DataFrame(hist),
    )

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Q1(b) GA Bit Pattern Generator", layout="wide")
st.title("Q1(b) Genetic Algorithm — Bit Pattern Generator")

st.markdown(
    f"""
**Fixed parameters (from question):**
- Population = **{POPULATION_SIZE}**
- Chromosome length = **{CHROMOSOME_LENGTH}**
- Generations = **{GENERATIONS}**
- Fitness peaks at ones = **{FITNESS_PEAK_ONES}**
- Max fitness = **{MAX_FITNESS}**
"""
)

with st.sidebar:
    st.header("GA Controls")
    elitism_n = st.slider("Elitism (keep best N)", 1, 20, 5)
    p_cross = st.slider("Crossover probability", 0.0, 1.0, 0.9)
    p_mut = st.slider("Mutation probability", 0.0, 0.2, 0.01)

run = st.button("Run Genetic Algorithm")

if run:
    result = run_ga(
        pop_size=POPULATION_SIZE,
        length=CHROMOSOME_LENGTH,
        generations=GENERATIONS,
        elitism_n=elitism_n,
        p_cross=p_cross,
        p_mut=p_mut,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Fitness", result.best_fitness)
    c2.metric("Ones in Best Chromosome", result.best_ones)
    c3.metric("Target Ones", FITNESS_PEAK_ONES)

    st.subheader("Best Bit Pattern (80 bits)")
    bit_string = "".join(map(str, result.best_bits.tolist()))
    st.code(bit_string)

    st.subheader("GA Progress")
    st.dataframe(result.history, use_container_width=True)

    fig = plt.figure()
    plt.plot(result.history["generation"], result.history["best_fitness"], label="Best fitness")
    plt.plot(result.history["generation"], result.history["avg_fitness"], label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    st.pyplot(fig)

    st.subheader("Interpretation (paste into report)")
    interp = f"""
- The fitness function is designed to **peak when the chromosome contains exactly {FITNESS_PEAK_ONES} ones**.
- Over generations, **best fitness increases** because selection favors individuals closer to the target ones-count.
- **Elitism** prevents losing the best solution, while **crossover** mixes useful gene patterns and **mutation** maintains diversity.
- The final best chromosome achieved **fitness = {result.best_fitness}** with **{result.best_ones} ones**.
- If the best ones-count is close to {FITNESS_PEAK_ONES}, it shows the GA successfully converged toward the optimum under the fixed parameters.
"""
    st.write(interp)

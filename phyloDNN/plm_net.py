from functools import lru_cache
from torch import adaptive_avg_pool1d, layer_norm, nn

# Load model directly
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_GENETIC_DIST = 8

LETTERS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
K = len(LETTERS)


class GPTDist(object):

    def __init__(self, repititions=1):
        self.tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.model = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2").cuda()
        self.repititions = repititions
        self.traces = []

    @lru_cache(maxsize=1028)
    def calculate_probability(self, x: str, i=None):
        """cache joint probability of observed sequence"""
        inputs = self.tokenizer(x, return_tensors="pt").to("cuda")
        log_probs = (
            self.model(**inputs, labels=inputs["input_ids"])
            .logits[0]
            .log_softmax(dim=1)
            .detach()
        )
        if i is not None:
            return log_probs[i].cpu()  # return log probs for site i
        return log_probs.gather(dim=1, index=inputs.input_ids).sum().cpu().item()

    def dist(self, x, y):
        """for infinite sites model, only allow one mutation per site, don't need MCMC.
        If a step is more costly"""
        x_log_prob = self.calculate_probability(x)
        L = len(x)
        distances = np.empty(self.repititions)
        for rep in range(self.repititions):
            steps = muts = 0
            x_t = x
            while x_t != y:
                steps += 1
                # assumes at most one mut/generation regardless of seq length
                available = [i for i in range(L) if x_t[i] != y[i]]
                # i = np.random.randint(L)  # takes forever
                i = np.random.choice(available)
                x_new = x_t[:i] + y[i] + x[i + 1 :]
                x_new_log_prob = self.calculate_probability(x_new)
                u = torch.rand(1).log().item()
                if u < x_new_log_prob - x_log_prob:
                    x_log_prob = x_new_log_prob
                    x_t = x_new
                    muts += 1
            distances[rep] = steps / L
        return distances

    def hamming(self, x, y):
        """Hamming distance between two sequences"""
        return sum(x[i] != y[i] for i in range(len(x)))

    def jc_dist(self, x, y):
        L = len(x)
        p = self.hamming(x, y) / L
        return -3 * np.log(1 - 4 / 3 * p) / 4 * L  # Jukes-Cantor distance

    def shortest_path_prob(self, x, y):
        """for infinite sites model, only allow one mutation per site, don't need MCMC.
        If a step is more costly"""
        x_log_prob = self.calculate_probability(x)
        L = len(x)
        distances = np.repeat(x_log_prob, self.repititions)
        mismatches = [i for i in range(L) if x[i] != y[i]]
        for rep in range(self.repititions):
            available = mismatches.copy()
            x_t = x
            while available:
                # assumes at most one mut/generation regardless of seq length
                i = np.random.choice(available)
                available.remove(i)
                x_t = x_t[:i] + y[i] + x_t[i + 1 :]
                x_new_log_prob = self.calculate_probability(x_t)
                distances[rep] += x_new_log_prob
        return distances

    def matrix_dist(self, x):
        """for infinite sites model, only allow one mutation per site, don't need MCMC.
        but need to figure out correction for multiple hits"""
        x_log_prob = self.calculate_probability(x)
        N, L = x.shape
        distances = np.empty(self.repititions)
        for rep in range(self.repititions):
            steps = muts = np.zeros(N)
            while len(x.unique(0)) > 1:
                steps += 1
                # assumes at most one mut/generation regardless of seq length
                taxon = np.random.randint(N)
                avail = x.uni
                available = [i for i in range(L) if x[i] != y[i]]
                # i = np.random.randint(L)  # takes forever
                i = np.random.choice(available)
                x_new = x[:i] + y[i] + x[i + 1 :]
                x_new_log_prob = self.calculate_probability(x_new)
                u = torch.rand(1).log().item()
                if u < x_new_log_prob - x_log_prob:
                    x_log_prob = x_new_log_prob
                    x = x_new
                    muts += 1
            distances[rep] = muts / steps / L
        return distances

    def prob_dist(self, x, y):
        """for finite sites model need MCMC"""
        x_log_prob = self.calculate_probability(x)
        shortest_prob = self.shortest_path_prob(x, y).max()
        m_obs = self.hamming(x, y)

        L = len(x)
        distances = np.repeat(m_obs, self.repititions)
        for rep in range(self.repititions):
            steps = 0
            prob = x_log_prob
            x_t = x
            while x_t != y:
                # assumes at most one mut/generation regardless of seq length
                # available = [i for i in range(L) if x_t[i] != y[i]]
                i = np.random.randint(L)  # takes forever
                # i = np.random.choice(available)
                s = LETTERS[np.random.randint(K)]
                if s == x_t[i]:
                    continue
                steps += 1
                x_t = x_t[:i] + s + x_t[i + 1 :]
                x_new_log_prob = self.calculate_probability(x_t)
                prob += x_new_log_prob
                if steps > m_obs and prob < shortest_prob:
                    break
            if prob > shortest_prob:
                distances[rep] = steps / L
        return distances

    def mcmc_dist(self, x, y):
        """for finite sites model need MCMC"""
        x_log_prob = self.calculate_probability(x)
        L = len(x)
        distances = np.empty(self.repititions)
        for rep in range(self.repititions):
            steps = muts = 0
            trace = [x_log_prob]
            x_t = x
            while x_t != y:
                steps += 1
                # assumes at most one mut/generation regardless of seq length
                # available = [i for i in range(L) if x_t[i] != y[i]]
                i = np.random.randint(L)  # takes forever
                # i = np.random.choice(available)
                s = LETTERS[np.random.randint(K)]
                if s == x[i]:
                    continue
                x_new = x_t[:i] + s + x[i + 1 :]
                x_new_log_prob = self.calculate_probability(x_new)
                trace.append(x_new_log_prob)
                u = torch.rand(1).log().item()
                if u <= x_new_log_prob - x_log_prob:
                    x_log_prob = x_new_log_prob
                    x_t = x_new
                    muts += 1
            self.traces.append(trace)
            distances[rep] = muts / L
        return distances


if __name__ == "__main__":
    x = "PSHKSLKIKRHLAKKQNQNRPIPNWIRLRTGNTIRYNAKRRHWRRTKLNL"
    y = "PAHKSFKIKVKLAKKMKQNRPIPQWVRLRTGNNIRYNAKRRHWRRTKLGL"
    m = GPTDist(repititions=5)
    d = m.prob_dist(x, y)
    djc = m.jc_dist(x, y)
    L = len(x)
    print(d, d.mean(), djc)

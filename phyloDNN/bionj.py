import numpy as np
import time
from dendropy import TaxonNamespace

class BioNJ:
    def __init__(self, precision=8, name_length=1000):
        """
        Initializes the BioNJ algorithm parameters.

        Args:
            precision (int): Precision for branch-lengths (not directly used in Python's float printing,
                             but kept for conceptual consistency).
            name_length (int): Maximum length of taxon names.
        """
        self.PREC = precision
        self.LEN = name_length

    def _initialize(self, distances, names):
        """
        Reads the input file and initializes the delta matrix and taxon names.
        The input file is expected to be in PHYLIP format.
        The delta matrix stores dissimilarities in the lower triangle and variances in the upper triangle.
        The first column (index 0) of delta is used to mark emptied rows.
        The diagonal of delta is used to store sums Sx.

        Args:
            input matrix: n x n
            n (int): Number of taxa.

        Returns:
            tuple: A tuple containing:
                - delta (np.ndarray): The initialized delta matrix.
                - trees (list): A list of taxon names.
        """
        n = len(names)
        # delta matrix: +1 for 1-based indexing in C, +1 for the 0th column used for 'emptied' flag
        delta = np.zeros((n + 1, n + 1), dtype=float)
        delta[1:, 1:] = distances
        if isinstance(names,TaxonNamespace):
            names=names.labels()
        trees = [''] + names

        return delta, trees

    def _symmetrize(self, delta, n):
        """
        Verifies if the delta matrix is symmetric and makes it symmetric if not.
        The rule used is Dij = Dji = (Dij + Dji) / 2.

        Args:
            delta (np.ndarray): The delta matrix.
            n (int): Number of taxa.

        Returns:
            bool: True if the matrix was initially symmetric or successfully symmetrized, False otherwise.
        """
        symmetric = True
        for lig in range(1, n + 1):
            for col in range(1, lig):  # Only iterate lower triangle
                if delta[lig][col] != delta[col][lig]:
                    value = (delta[lig][col] + delta[col][lig]) / 2.0
                    delta[lig][col] = value
                    delta[col][lig] = value
                    symmetric = False
        if not symmetric:
            print("The matrix was not symmetric and has been symmetrized.")
        return symmetric

    def _get_distance(self, i, j, delta):
        """
        Retrieves the distance between taxa i and j from the delta matrix.
        Assumes distances are stored in the lower triangle.
        """
        if i > j:
            return delta[i][j]
        else:
            return delta[j][i]

    def _get_variance(self, i, j, delta):
        """
        Retrieves the variance of the distance between i and j from the delta matrix.
        Assumes variances are stored in the upper triangle.
        """
        if i > j:
            return delta[j][i]
        else:
            return delta[i][j]

    def _is_emptied(self, i, delta):
        """
        Checks if a line (subtree) is marked as emptied.
        A value of 1.0 in delta[i][0] indicates an emptied line.
        """
        return delta[i][0] == 1.0

    def _get_sum_s(self, i, delta):
        """
        Retrieves the sum Sx from the diagonal of the delta matrix.
        """
        return delta[i][i]

    def _compute_sums_sx(self, delta, n):
        """
        Computes the sums Sx for each active taxon and stores them in the
        diagonal of the delta matrix.
        """
        for i in range(1, n + 1):
            if not self._is_emptied(i, delta):
                s_sum = 0.0
                for j in range(1, n + 1):
                    if i != j and not self._is_emptied(j, delta):
                        s_sum += self._get_distance(i, j, delta)
                delta[i][i] = s_sum

    def _agglomerative_criterion(self, i, j, delta, r):
        """
        Calculates the agglomerative criterion (Formula 1).
        Q_ij = (r - 2) * D_ij - S_i - S_j
        """
        q_ij = (r - 2) * self._get_distance(i, j, delta) - \
            self._get_sum_s(i, delta) - \
            self._get_sum_s(j, delta)
        return q_ij

    def _best_pair(self, delta, r, n):
        """
        Finds the best pair to be agglomerated by minimizing the
        agglomerative criterion.

        Args:
            delta (np.ndarray): The delta matrix.
            r (int): Current number of active subtrees.
            n (int): Total number of taxa.

        Returns:
            tuple: A tuple (a, b) representing the indices of the best pair.
        """
        q_min = float('inf')
        best_a = -1
        best_b = -1

        for x in range(1, n + 1):
            if not self._is_emptied(x, delta):
                for y in range(1, x):  # Iterate through lower triangle to avoid duplicates
                    if not self._is_emptied(y, delta):
                        q_xy = self._agglomerative_criterion(x, y, delta, r)
                        if q_xy < q_min - 1e-7:  # Using a small epsilon for float comparison
                            q_min = q_xy
                            best_a = x
                            best_b = y
        return best_a, best_b

    def _branch_length(self, a, b, delta, r):
        """
        Computes the branch length using Formula (2).
        L_a = 0.5 * (D_ab + (S_a - S_b) / (r - 2))
        """
        length = 0.5 * (self._get_distance(a, b, delta) +
                        (self._get_sum_s(a, delta) - self._get_sum_s(b, delta)) / (r - 2))
        return length

    def _lamda(self, a, b, vab, delta, n, r):
        """
        Computes lambda* using Formula (9) and applies the constraint that lambda is in [0, 1].
        """
        lamda_val = 0.0
        if vab == 0.0:
            lamda_val = 0.5
        else:
            sum_diff_variances = 0.0
            for i in range(1, n + 1):
                if i != a and i != b and not self._is_emptied(i, delta):
                    sum_diff_variances += (self._get_variance(b,
                                           i, delta) - self._get_variance(a, i, delta))
            lamda_val = 0.5 + sum_diff_variances / (2 * (r - 2) * vab)

        # Apply constraints for lambda
        return max(min(lamda_val,1),0)

    def _reduction_d(self, a, la, b, lb, i, lamda, delta):
        """
        Applies Reduction Formula (4) for distances.
        D_ui = lambda * (D_ai - L_a) + (1 - lambda) * (D_bi - L_b)
        """
        d_ui = lamda * (self._get_distance(a, i, delta) - la) + \
            (1 - lamda) * (self._get_distance(b, i, delta) - lb)
        return d_ui

    def _reduction_v(self, a, b, i, lamda, vab, delta):
        """
        Applies Reduction Formula (10) for variances.
        V_ci = lambda * V_ai + (1 - lambda) * V_bi - lambda * (1 - lambda) * V_ab
        """
        v_ci = lamda * self._get_variance(a, i, delta) + \
            (1 - lamda) * self._get_variance(b, i, delta) - \
            lamda * (1 - lamda) * vab
        return v_ci

    def _concatenate(self, new_string, ind, trees, post):
        """
        Concatenates a string to a subtree representation.
        In Python, we'll directly manipulate the string representation.
        This function is adapted for Python string concatenation.
        """
        if post == 0:  # Prepend
            trees[ind] = new_string + trees[ind]
        else:  # Append
            trees[ind] = trees[ind] + new_string

    def _finish_branch_length(self, i, j, k, delta):
        """
        Computes the length of the branch attached to subtree i during the final step.
        """
        length = 0.5 * (self._get_distance(i, j, delta) + self._get_distance(i, k, delta) -
                        self._get_distance(j, k, delta))
        return length

    def _print_output(self, i, trees, output_file):
        """
        Prints the subtree (taxon name or Newick string) to the output file.
        """
        output_file.write(trees[i])

    def reconstruct_tree(self, input_distances, input_names, verbose=False):
        """
        Main function to reconstruct the phylogenetic tree using BioNJ.

        Args:
            input_filepath (str): Path to the input file (PHYLIP format).
            output_filepath (str): Path to the output file (Newick format).
        """
        clock_start = time.time()
        n = len(input_names)
        delta, trees = self._initialize(input_distances, input_names)
        self._symmetrize(delta, n)

        r = n  # Current number of active subtrees

        # Main BioNJ loop
        while r > 3:
            self._compute_sums_sx(delta, n)
            a, b = self._best_pair(delta, r, n)

            v_ab = self._get_variance(a, b, delta)
            l_a = self._branch_length(a, b, delta, r)
            l_b = self._branch_length(b, a, delta, r)
            lamda_val = self._lamda(a, b, v_ab, delta, n, r)

            # Update delta matrix
            for i in range(1, n + 1):
                if not self._is_emptied(i, delta) and i != a and i != b:
                    # Update distance (lower triangle)
                    # The C code uses `x` and `y` to ensure delta[x][y] is always lower triangle
                    # We can use min/max or directly assign since we call _reduction_d
                    # and then store it in delta[max(a,i)][min(a,i)] after calculation
                    # The C code seems to update delta[x][y] as D_ui and delta[y][x] as V_ci
                    # Let's clarify this
                    # In C:
                    # delta[x][y]=Reduction4(*a, la, *b, lb, i, lamda, delta); // Lower triangle for D_ui
                    # delta[y][x]=Reduction10(*a, *b, i, lamda, vab, delta); // Upper triangle for V_ci

                    # In Python, we calculate new D_ui and V_ci and then place them correctly
                    new_d_ui = self._reduction_d(
                        a, l_a, b, l_b, i, lamda_val, delta)
                    new_v_ci = self._reduction_v(
                        a, b, i, lamda_val, v_ab, delta)

                    # Update delta matrix for the new node (u = a + b)
                    # The new node effectively replaces 'a'. 'b' is marked as emptied.
                    # The new distances and variances are between 'a' (now representing the merged node) and 'i'
                    min_idx = min(a, i)
                    max_idx = max(a, i)
                    delta[max_idx][min_idx] = new_d_ui  # Distance
                    delta[min_idx][max_idx] = new_v_ci  # Variance

            # Agglomerate subtrees and update Newick string
            # Example: (taxonA:lengthA,taxonB:lengthB)
            temp_a_str = trees[a]
            temp_b_str = trees[b]

            trees[a] = f"({temp_a_str}:{l_a:.{self.PREC}f},{temp_b_str}:{l_b:.{self.PREC}f})"

            # Mark 'b' as emptied
            delta[b][0] = 1.0
            trees[b] = ""  # Clear the string representation for 'b'

            r -= 1  # Decrease number of active subtrees

        # Final step: process the last three remaining subtrees
        last_three_indices = [idx for idx in range(
            1, n + 1) if not self._is_emptied(idx, delta)]
        output = ''
        if len(last_three_indices) == 3:
            i, j, k = last_three_indices[0], last_three_indices[1], last_three_indices[2]

            len_i = self._finish_branch_length(i, j, k, delta)
            len_j = self._finish_branch_length(j, i, k, delta)
            len_k = self._finish_branch_length(k, i, j, delta)

            output += f"({trees[i]}:{len_i:.{self.PREC}f},"
            output += f"{trees[j]}:{len_j:.{self.PREC}f},"
            output += f"{trees[k]}:{len_k:.{self.PREC}f});\n"
        else:
            # Handle cases where r might be less than 3 at the start (e.g., n=2)
            # This simplified BioNJ variant expects n >= 3 for the loop.
            # If n=2, it would simply be (taxon1:d1,taxon2:d2)
            if n == 2:
                output += f"({trees[1]}:{delta[1][2]:.{self.PREC}f},{trees[2]}:{delta[1][2]:.{self.PREC}f});\n"
            elif n < 3:
                # For n=1, it's just the taxon name
                # Check for the single remaining node
                if not self._is_emptied(1, delta):
                    output += f"{trees[1]};\n"
                else:
                    print("Error: No taxa to process.")
            else:
                print(
                    f"Unexpected number of remaining subtrees: {len(last_three_indices)}. Expected 3.")

        if verbose:
            clock_end = time.time()
            total_time = clock_end - clock_start
            print(f"Time: {total_time:.3f} seconds")
        return output
    
# import dendropy
# import math
# import copy

# def bionj(taxon_names, distance_matrix, variance_matrix=None):
#     """
#     Computes a phylogenetic tree using Gascuel's BIONJ algorithm.

#     Args:
#         taxon_names (list): A list of strings, where each string is a taxon name.
#         distance_matrix (list of lists): A square, symmetric 2D list representing
#                                          the pairwise evolutionary distances between taxa.
#         variance_matrix (list of lists, optional): A square, symmetric 2D list
#                                                    representing the variances of the
#                                                    pairwise distances. If None,
#                                                    all variances are assumed to be 1.0.

#     Returns:
#         dendropy.Tree: A DendroPy Tree object representing the inferred phylogeny.

#     Raises:
#         ValueError: If input dimensions are incorrect or less than two taxa are provided.
#     """
#     num_taxa = len(taxon_names)
#     if num_taxa < 2:
#         raise ValueError("BIONJ algorithm requires at least two taxa.")
#     if len(distance_matrix) != num_taxa or any(len(row) != num_taxa for row in distance_matrix):
#         raise ValueError("Distance matrix must be square and its dimensions must match the number of taxa.")
#     if variance_matrix is not None and (len(variance_matrix) != num_taxa or any(len(row) != num_taxa for row in variance_matrix)):
#         raise ValueError("Variance matrix must be square and its dimensions must match the number of taxa.")

#     # 1. Initialization
#     # Create a TaxonNamespace for DendroPy to manage taxon objects
#     taxon_namespace = dendropy.TaxonNamespace(taxon_names)

#     # Create initial leaf nodes for each taxon
#     active_nodes = []
#     # Map original taxon index to its corresponding DendroPy Node object
#     initial_node_map = {}
#     for i, name in enumerate(taxon_names):
#         node = dendropy.Node(taxon=taxon_namespace.get_taxon(name))
#         active_nodes.append(node)
#         initial_node_map[i] = node

#     # Initialize current distance and variance matrices as dictionaries for efficient lookup and updates.
#     # Keys are tuples of (node1, node2), ensuring consistent order by using object IDs.
#     current_distances = {}
#     current_variances = {}

#     # Populate initial distances and variances from the input matrices
#     for i in range(num_taxa):
#         for j in range(i + 1, num_taxa): # Only iterate upper triangle since matrices are symmetric
#             node_i = initial_node_map[i]
#             node_j = initial_node_map[j]

#             # Use sorted(..., key=id) to ensure a consistent key order for dictionary lookups
#             # (e.g., (node_A, node_B) vs (node_B, node_A) will map to the same key)
#             key = tuple(sorted((node_i, node_j), key=id))

#             current_distances[key] = distance_matrix[i][j]
#             # Also store the symmetric entry for convenience, though not strictly necessary with sorted keys
#             # current_distances[tuple(sorted((node_j, node_i), key=id))] = distance_matrix[j][i]

#             if variance_matrix is not None:
#                 current_variances[key] = variance_matrix[i][j]
#                 # current_variances[tuple(sorted((node_j, node_i), key=id))] = variance_matrix[j][i]
#             else:
#                 # Default variance: constant 1.0 if not provided
#                 current_variances[key] = 1.0
#                 # current_variances[tuple(sorted((node_j, node_i), key=id))] = 1.0

#     # Keep track of all nodes created (leaves and internal) to link them correctly
#     all_nodes = list(active_nodes)
#     next_internal_node_idx = 0 # Counter for naming internal nodes

#     # 2. Iteration: Continue until only two active nodes remain
#     while len(active_nodes) > 2:
#         N = len(active_nodes) # Current number of active nodes

#         # Calculate r_i for each active node i
#         # r_i is the sum of distances from node i to all other active nodes
#         r_values = {}
#         for i_node in active_nodes:
#             r_i_sum = 0.0
#             for k_node in active_nodes:
#                 if i_node != k_node:
#                     # Retrieve distance using the consistent key order
#                     key = tuple(sorted((i_node, k_node), key=id))
#                     r_i_sum += current_distances[key]
#             r_values[i_node] = r_i_sum

#         # Calculate Q_ij for all unique pairs (i, j) and find the pair with the minimum Q_ab
#         min_q = float('inf')
#         pair_to_join = None

#         # Iterate through all unique pairs of active nodes
#         for i_idx in range(N):
#             for j_idx in range(i_idx + 1, N): # Avoid duplicate pairs and self-pairs
#                 node_i = active_nodes[i_idx]
#                 node_j = active_nodes[j_idx]

#                 # Get the distance between node_i and node_j
#                 key_ij = tuple(sorted((node_i, node_j), key=id))
#                 d_ij = current_distances[key_ij]

#                 # Calculate Q_ij using the Neighbor-Joining formula
#                 # Q_ij = D_ij - (r_i + r_j) / (N - 2)
#                 q_ij = d_ij - (r_values[node_i] + r_values[node_j]) / (N - 2)

#                 # Find the pair that minimizes Q_ij
#                 if q_ij < min_q:
#                     min_q = q_ij
#                     pair_to_join = (node_i, node_j)

#         # Extract the two nodes to be joined
#         a, b = pair_to_join

#         # Create a new internal node 'u' that will be the parent of 'a' and 'b'
#         u = dendropy.Node(label=f"internal_{next_internal_node_idx}")
#         next_internal_node_idx += 1
#         all_nodes.append(u) # Add the new internal node to the list of all nodes

#         # Calculate branch lengths from the new node 'u' to its children 'a' and 'b'
#         # These formulas are derived from the NJ/BIONJ algorithm
#         key_ab = tuple(sorted((a, b), key=id))
#         d_ab = current_distances[key_ab]
        
        
        
#         d_au = (d_ab + (r_values[a] - r_values[b]) / (N - 2)) / 2
#         d_bu = (d_ab + (r_values[b] - r_values[a]) / (N - 2)) / 2

#         # Assign branch lengths, ensuring they are non-negative
#         a.edge_length = max(0.0, d_au)
#         b.edge_length = max(0.0, d_bu)

#         # Attach 'a' and 'b' as children of 'u'
#         u.add_child(a)
#         u.add_child(b)

#         # Update distances and variances for the new node 'u' with respect to all other active nodes 'k'
#         new_distances_for_u = {}
#         new_variances_for_u = {}
        
#         v = 0
#         for k_node in active_nodes:
#             if k_node != a and k_node != b: # For all nodes k that are not a or b
#                 # Retrieve distances and variances involving a, b, and k
#                 key_ak = tuple(sorted((a, k_node), key=id))
#                 key_bk = tuple(sorted((b, k_node), key=id))
#                 v_ak = current_variances[key_ak]
#                 v_bk = current_variances[key_bk]
#                 v += v_ak - v_bk
#         v_ab = current_variances[key_ab] # Variance of the joined pair (a,b)
#         lam = .5 + v/( (N-2) * 2 * v_ab)
#         lam = min(max(lam,0),1)
#         # print(lam)
        
                
#         for k_node in active_nodes:
#             if k_node != a and k_node != b: # For all nodes k that are not a or b
#                 # Retrieve distances and variances involving a, b, and k
#                 key_ak = tuple(sorted((a, k_node), key=id))
#                 key_bk = tuple(sorted((b, k_node), key=id))

#                 d_ak = current_distances[key_ak]
#                 d_bk = current_distances[key_bk]
#                 v_ak = current_variances[key_ak]
#                 v_bk = current_variances[key_bk]
                

#                 # Calculate the new distance from 'u' to 'k'
#                 # D_uk = (D_ak + D_bk - D_ab) / 2
#                 d_uk = (lam * d_ak + (1-lam) * d_bk - lam*d_au - (1-lam)*d_bu) / 2

#                 # Calculate the new variance for D_uk (BIONJ's specific variance propagation)
#                 # Var(D_uk) = (Var(D_ak) + Var(D_bk) + Var(D_ab)) / 2
#                 v_uk = (lam * v_ak + (1-lam) * v_bk -  lam*(1-lam)*v_ab) / 2

#                 # Store the new distance and variance with 'u'
#                 key_uk = tuple(sorted((u, k_node), key=id))
#                 new_distances_for_u[key_uk] = d_uk
#                 new_variances_for_u[key_uk] = v_uk

#         # Update the global distance and variance dictionaries
#         # First, remove all entries involving 'a' or 'b' as they are now merged into 'u'
#         keys_to_remove = []
#         for k1, k2 in current_distances.keys():
#             if k1 in (a, b) or k2 in (a, b):
#                 keys_to_remove.append(tuple(sorted((k1, k2), key=id)))

#         for key in set(keys_to_remove): # Use a set to handle potential duplicate keys if they were added
#             if key in current_distances:
#                 del current_distances[key]
#             if key in current_variances:
#                 del current_variances[key]

#         # Then, add the new entries involving 'u'
#         current_distances.update(new_distances_for_u)
#         current_variances.update(new_variances_for_u)

#         # Update the list of active nodes: remove 'a' and 'b', add 'u'
#         active_nodes.remove(a)
#         active_nodes.remove(b)
#         active_nodes.append(u)

#     # 3. Final Step: When only two nodes remain, connect them to the root
#     node1, node2 = active_nodes[0], active_nodes[1]

#     # Create the final root node for the tree
#     root = dendropy.Node(label=f"root_{next_internal_node_idx}")
#     all_nodes.append(root)

#     # The distance between the last two nodes is split equally to form their branch lengths to the root
#     key_12 = tuple(sorted((node1, node2), key=id))
#     final_dist = current_distances[key_12]

#     node1.edge_length = max(0.0, final_dist / 2.0)
#     node2.edge_length = max(0.0, final_dist / 2.0)

#     # Attach the last two nodes as children of the root
#     root.add_child(node1)
#     root.add_child(node2)

#     # Create the DendroPy tree object
#     tree = dendropy.Tree(seed_node=root, taxon_namespace=taxon_namespace)

#     return tree
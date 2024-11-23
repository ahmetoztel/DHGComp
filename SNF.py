from PIL import Image
import numpy as np
from tkinter import filedialog
from tkinter import Tk
import os
from scipy.sparse import lil_matrix
from sympy import Matrix


# Function to select an image file using a file dialog
def select_image_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("All Image Files", "*.*")])
    return file_path


# Converts an image into a binary matrix (black and white)
def image_to_binary_matrix(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None, 0
    try:
        # Convert image to grayscale
        image = Image.open(file_path).convert('L')
        image_np = np.array(image)
        # Thresholding based on the mean value
        threshold = np.mean(image_np)
        binary_matrix = np.where(image_np > threshold, 1, 0)
        black_pixel_count = np.sum(binary_matrix == 0)  # Count the black pixels
        print(f"Detected black pixel count: {black_pixel_count}")
        return binary_matrix, black_pixel_count
    except Exception as e:
        print(f"Error: Failed to read the image. {e}")
        return None, 0


# Finds the neighbors of each black pixel in the binary matrix
def find_black_pixel_neighbors(binary_matrix):
    neighbors = {}
    rows, cols = binary_matrix.shape
    for y in range(rows):
        for x in range(cols):
            if binary_matrix[y, x] == 0:  # Only consider black pixels
                pixel_neighbors = []
                for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        if binary_matrix[ny, nx] == 0:
                            pixel_neighbors.append((nx, ny))
                neighbors[(x, y)] = pixel_neighbors
    return neighbors


# Generates simplices (points, edges, triangles, and tetrahedra) from black pixels
def generate_simplices(binary_matrix):
    black_pixel_neighbors = find_black_pixel_neighbors(binary_matrix)
    zero_simplices = []
    one_simplices = set()
    two_simplices = set()
    three_simplices = set()

    # Generate 0-simplices (individual black pixels)
    for pixel in black_pixel_neighbors.keys():
        zero_simplices.append(pixel)

    # Generate 1-simplices (edges between neighboring black pixels)
    for pixel, neighbors in black_pixel_neighbors.items():
        for neighbor in neighbors:
            if pixel < neighbor:
                one_simplices.add((pixel, neighbor))

    # Generate 2-simplices (triangles formed by three connected black pixels)
    for pixel, neighbors in black_pixel_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in black_pixel_neighbors[neighbors[i]]:
                    v0, v1, v2 = sorted([pixel, neighbors[i], neighbors[j]])
                    two_simplices.add((v0, v1, v2))

    # Generate 3-simplices (tetrahedra formed by four connected black pixels)
    for pixel, neighbors in black_pixel_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    v0, v1, v2, v3 = sorted([pixel, neighbors[i], neighbors[j], neighbors[k]])
                    if (
                            v1 in black_pixel_neighbors[v0] and
                            v2 in black_pixel_neighbors[v0] and
                            v3 in black_pixel_neighbors[v0] and
                            v2 in black_pixel_neighbors[v1] and
                            v3 in black_pixel_neighbors[v1] and
                            v3 in black_pixel_neighbors[v2]
                    ):
                        three_simplices.add((v0, v1, v2, v3))

    return zero_simplices, list(one_simplices), list(two_simplices), list(three_simplices)


# Generates sparse boundary matrices for simplices
def generate_boundary_matrices_sparse(zero_simplices, one_simplices, two_simplices, three_simplices):
    zero_simplex_indices = {simplex: i for i, simplex in enumerate(zero_simplices)}
    one_simplex_indices = {simplex: i for i, simplex in enumerate(one_simplices)}
    two_simplex_indices = {simplex: i for i, simplex in enumerate(two_simplices)}
    three_simplex_indices = {simplex: i for i, simplex in enumerate(three_simplices)}

    # Boundary matrix B0 (no boundaries for 0-simplices)
    B0 = lil_matrix((len(zero_simplices), 1), dtype=int)

    # Boundary matrix B1 (edges to points)
    B1 = lil_matrix((len(one_simplices), len(zero_simplices)), dtype=int)
    for one_simplex in one_simplices:
        v0, v1 = one_simplex
        row = one_simplex_indices[one_simplex]
        B1[row, zero_simplex_indices[v0]] = 1
        B1[row, zero_simplex_indices[v1]] = -1

    # Boundary matrix B2 (triangles to edges)
    B2 = lil_matrix((len(two_simplices), len(one_simplices)), dtype=int)
    for two_simplex in two_simplices:
        v0, v1, v2 = two_simplex
        edges = [(v0, v1), (v1, v2), (v0, v2)]
        row = two_simplex_indices[two_simplex]
        for i, edge in enumerate(edges):
            col = one_simplex_indices[edge]
            sign = 1 if edge == (v0, v1) or edge == (v1, v2) else -1
            B2[row, col] = sign

    # Boundary matrix B3 (tetrahedra to triangles)
    B3 = lil_matrix((len(three_simplices), len(two_simplices)), dtype=int)
    for three_simplex in three_simplices:
        v0, v1, v2, v3 = three_simplex
        faces = [(v0, v1, v2), (v0, v1, v3), (v0, v2, v3), (v1, v2, v3)]
        row = three_simplex_indices[three_simplex]
        for i, face in enumerate(faces):
            col = two_simplex_indices[tuple(sorted(face))]
            sign = 1 if face == (v0, v1, v2) or face == (v0, v2, v3) else -1
            B3[row, col] = sign

    return B0, B1, B2, B3


# Approximates the Smith normal form of a matrix
def approximate_smith_normal_form(matrix):
    sympy_matrix = Matrix(matrix)
    row_echelon, _ = sympy_matrix.rref()  # Row echelon form
    transposed = row_echelon.transpose()
    row_echelon_transposed, _ = transposed.rref()
    smith_approx = row_echelon_transposed.transpose()
    return smith_approx


# Counts zero rows and non-zero columns in a matrix
def count_zero_rows_columns(matrix):
    zero_rows = sum(1 for row in matrix.tolist() if all(val == 0 for val in row))
    non_zero_columns = sum(1 for col in zip(*matrix.tolist()) if any(val != 0 for val in col))
    return zero_rows, non_zero_columns


# Calculates Betti numbers and Euler characteristic
def calculate_betti_numbers_and_euler_characteristic(B0, B1, B2, B3, zero_simplices, one_simplices, two_simplices,
                                                     three_simplices, black_pixel_count):
    snf_B0 = approximate_smith_normal_form(B0.toarray())
    snf_B1 = approximate_smith_normal_form(B1.toarray())
    snf_B2 = approximate_smith_normal_form(B2.toarray())
    snf_B3 = approximate_smith_normal_form(B3.toarray())

    zero_rows_B0, _ = count_zero_rows_columns(snf_B0)
    zero_rows_B1, non_zero_columns_B1 = count_zero_rows_columns(snf_B1)
    zero_rows_B2, non_zero_columns_B2 = count_zero_rows_columns(snf_B2)
    zero_rows_B3, non_zero_columns_B3 = count_zero_rows_columns(snf_B3)

    beta_0 = zero_rows_B0 - non_zero_columns_B1
    beta_1 = zero_rows_B1 - non_zero_columns_B2
    beta_2 = zero_rows_B2 - non_zero_columns_B3
    beta_3 = zero_rows_B3

    euler_betti = beta_0 - beta_1 + beta_2 - beta_3
    euler_simplices = len(zero_simplices) - len(one_simplices) + len(two_simplices) - len(three_simplices)
    euler_consistency = euler_betti == euler_simplices

    print(f"Betti Numbers:")
    print(f"Beta 0 (β₀): {beta_0}")
    print(f"Beta 1 (β₁): {beta_1}")
    print(f"Beta 2 (β₂): {beta_2}")
    print(f"Beta 3 (β₃): {beta_3}\n")
    print("Euler Characteristic:")
    print(f"Euler Characteristic (from Betti numbers): {euler_betti}")
    print(f"Euler Characteristic (from simplices): {euler_simplices}")
    print(f"Euler Characteristic Consistency: {euler_consistency}")


# Main process
file_path = select_image_file()
binary_matrix, black_pixel_count = image_to_binary_matrix(file_path)
if binary_matrix is not None:
    zero_simplices, one_simplices, two_simplices, three_simplices = generate_simplices(binary_matrix)
    B0, B1, B2, B3 = generate_boundary_matrices_sparse(zero_simplices, one_simplices, two_simplices, three_simplices)

    calculate_betti_numbers_and_euler_characteristic(
        B0, B1, B2, B3, zero_simplices, one_simplices, two_simplices, three_simplices, black_pixel_count)
else:
    print("No file selected or error in processing.")

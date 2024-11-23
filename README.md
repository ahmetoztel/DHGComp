# DHGComp
# README.md

## Image to Simplicial Complex Converter and Topological Analysis

This Python program processes digital images to extract topological structures, such as simplicial complexes, and computes their Betti numbers and Euler characteristics. It leverages mathematical concepts from algebraic topology to analyze the shape and structure of black pixels in a binary image.

---

## Features

- **Image Processing**: Converts grayscale images into binary matrices based on pixel intensity.
- **Simplicial Complex Construction**: Generates 0-simplices (vertices), 1-simplices (edges), 2-simplices (triangles), and 3-simplices (tetrahedra) based on the connectivity of black pixels.
- **Boundary Matrices**: Constructs sparse boundary matrices to represent the relations between simplices.
- **Topological Invariants**:
  - Computes **Betti numbers** \( (\beta_0, \beta_1, \beta_2, \beta_3) \) to represent the rank of homology groups.
  - Calculates the **Euler characteristic** using both Betti numbers and simplex counts.
  - Verifies consistency between different methods of Euler characteristic computation.

---

## Dependencies

This program requires the following Python libraries:

- `Pillow`: For image processing.
- `NumPy`: For numerical matrix operations.
- `SciPy`: For sparse matrix representation.
- `SymPy`: For matrix manipulations and approximating Smith normal forms.
- `Tkinter`: For GUI-based file selection.

Install the required packages via pip:

```bash
pip install pillow numpy scipy sympy
```

---

## How to Use

1. **Run the Script**:
   Execute the Python script in your terminal or IDE.

   ```bash
   python script_name.py
   ```

2. **Select an Image**:
   - A file dialog will appear to let you choose an image file.
   - Supported formats include any image types compatible with the Pillow library (e.g., JPEG, PNG, BMP).

3. **Output**:
   - The program will display the following:
     - Number of detected black pixels.
     - Betti numbers for each homology group.
     - Euler characteristic values and their consistency check.

---

## Example Output

```
Detected black pixel count: 1024
Betti Numbers:
Beta 0 (β₀): 1
Beta 1 (β₁): 2
Beta 2 (β₂): 0
Beta 3 (β₃): 0

Euler Characteristic:
Euler Characteristic (from Betti numbers): -1
Euler Characteristic (from simplices): -1
Euler Characteristic Consistency: True
```

---

## Functions Overview

### 1. **Image to Binary Matrix**:
   - Converts the selected image into a binary matrix (1 for white, 0 for black).

### 2. **Generate Simplices**:
   - Identifies simplices (vertices, edges, triangles, and tetrahedra) based on black pixel connectivity.

### 3. **Boundary Matrices**:
   - Constructs sparse boundary matrices for simplicial complexes to enable topological calculations.

### 4. **Topological Invariants**:
   - Approximates Smith normal forms and calculates Betti numbers and Euler characteristics.

---

## Notes

- The script uses an approximate Smith normal form due to computational limitations.
- Euler characteristic consistency provides a quick sanity check for the correctness of calculations.

---

## License

This project is open-source and available under the MIT License.

---

Feel free to modify the script to adapt it to your specific needs or extend its functionality!

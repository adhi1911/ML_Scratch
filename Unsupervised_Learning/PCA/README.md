# Principal Component Analysis (PCA)

## Introduction

Principal Component Analysis (PCA) is a powerful technique used for dimensionality reduction and feature extraction. It allows us to transform high-dimensional data into a lower-dimensional space while preserving essential information. Let's dive into the key concepts step by step.

### Representation Learning

1. **Covariance Matrix:**
   - The covariance matrix captures the relationships between different features in our dataset.
   - Given *n* data points with *m* features, the covariance matrix \(C\) is computed as:
     $$ C = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T $$
     where \(x_i\) represents the \(i\)-th data point, and \(\bar{x}\) is the mean vector.

2. **Eigenvalue Decomposition:**
   - We compute the eigenvalues (\(\lambda_1, \lambda_2, \ldots, \lambda_m\)) and corresponding eigenvectors (\(v_1, v_2, \ldots, v_m\)) of the covariance matrix.
   - Eigenvectors represent the principal components (PCs), which are orthogonal directions in the feature space.
   - Eigenvalues indicate the variance explained by each PC.

### Intuition for PCA

1. **Selecting Principal Components:**
   - Sort the eigenvectors by their corresponding eigenvalues in descending order.
   - Choose the top-\(k\) eigenvectors (where \(k\) is the desired reduced dimensionality).
   - These top PCs capture the most significant variance in the data.

2. **Projection:**
   - Project the original data onto the selected principal components.
   - The reduced-dimensional representation is obtained by multiplying the data by the chosen eigenvectors:
     $$ \text{Reduced Data} = X \cdot V_k $$
     where \(X\) is the original data matrix, and \(V_k\) contains the top-\(k\) eigenvectors.

### Mathematical Expressions

Let's summarize the key mathematical expressions used in PCA:

1. **Representation Constraint**:
   - The linear representation is denoted by \(w\), subject to the constraint \(||w|| = 1\).

2. **Projection Formula**:
   - The projection of data point \(x_i\) on \(w\) is given by \((x_i^T w)w\).

3. **Reconstruction Error**:
   - The reconstruction error for a given \(w\) is calculated as:
     $$ \text{Reconstruction Error}(f(w)) = \frac{1}{n} \sum_{i=1}^{n} ||x_i - (x_i^T w)w||^2 $$

4. **Optimization Formulation**:
   - To minimize the reconstruction error, the problem is formulated as:
     $$ \max_{w \in ||w||=1} f(w) = w^T Cw $$
     where \(C\) is the Covariance Matrix.

## Algorithm Implementation

To implement PCA, follow these steps:

1. **Data Preprocessing:**
   - Standardize or normalize the input features.
   - Remove any mean or center the data.

2. **Compute Covariance Matrix:**
   - Calculate the covariance matrix based on the preprocessed data.

3. **Eigenvalue Decomposition:**
   - Compute eigenvalues and eigenvectors using matrix operations.
   - Sort them in descending order.

4. **Select Top-\(k\) Eigenvectors:**
   - Choose the first \(k\) eigenvectors corresponding to the highest eigenvalues.

5. **Project Data:**
   - Multiply the original data by the selected eigenvectors to obtain the reduced-dimensional representation.

## Usage and Examples

In this repository, you'll find an implementation of PCA in Python. Use the provided code to apply PCA to your datasets. Happy coding! 🚀
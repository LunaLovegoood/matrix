# Matrix

Hole **Matrix** class and it's relative classes are placed into the namespace matrix.
In the following samples, I'll suppose you've included **matrix.h** to your code and added the following line to your code:
```cpp
using namespace matrix;
```

### Creating an instance of Matrix

**Matrix** can be created in one of the following ways:
```cpp
// creates empty Matrix
Matrix default_matrix(); 
// creates Matrix with 2 rows and 3 columns filled with zeros
Matrix zero_matrix(2, 3); 
// creates Matrix with 2 rows and 3 columns filled with random values in between of lower_bound and upper_bound
// if lower_bound and upper_bound are omitted, then values will be generated in the interval [0;1]
Matrix random_matrix(2, 3, true, lower_bound, upper_bound);
// creates Matrix from std::vector of std::vectors
Matrix vect_matrix(vector_of_vectors);
```

There are also special square matrices, which can be created using static function-members of **Matrix** class:
```cpp
Matrix zeros = Matrix::zeros(5); // creates zero matrix of order 5
Matrix ones = Matrix::ones(5); // creates matrix of order 5 filled with ones
Matrix identity = Matrix::identity(5); // creates identity matrix of order 5
```

### Matrix operations

Here's samples of common matrix operations (shorthand assignment versions are also available):
```cpp
matrix + 5; // adds 5 to the matrix
matrix + another_matrix; // adds two matrices

matrix - 5; // subtracts 5 from the matrix
matrix - another_matrix; // subtracts another_matrix from matrix

matrix * 5; // multiplies matrix by 5
matrix * another_matrix; // multiplies matrix by another_matrix

matrix / 5; // divides matrix by 5
```

Other matrix operations can be performed using member-functions:
```cpp
matrix.hadm_product(another_matrix); // returns Hadamard product of two matrices
matrix.transpose(); // return transposed matrix
matrix.map(map_function); // applies map_function to every element of the matrix
```

All comparison operators except == are **deleted**.

### Additional functions

**Matrix** contains a few additional functions, which can be helpful in certain situations:
```cpp
matrix.merge(another_matrix); // returns matrix created with merging of two matrices
matrix.fill(number); // fills matrix with number
matrix.fill_random(lower_bound, upper_bound); // fill matrix with random number (by default in the interval [0,1])
matrix.set_element(row, col, new_value); // sets new_value to matrix element with indices row and col
matrix.copy(another_matrix); // deletes previous matrix and copies another_matrix to matrix
```

---

## Contributors

- Yurii Khomiak

---

## License & copyright

© Yurii Khomiak

Licensed under the [MIT License](LICENSE).
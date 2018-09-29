# Matrix

Hole Matrix class and it's relative classes are placed into the namespace matrix.
In the following samples, I'll suppose you've added next line to your code:
```cpp
using namespace matrix;
```

### Creating instance of Matrix

Matrix can be created in one of the following ways:
```cpp
// creates empty Matrix
Matrix default_matrix(); 
// creates Matrix with 2 rows and 3 columns filled with zeros
Matrix zero_matrix(2, 3); 
// creates Matrix with 2 rows and 3 columns filled with random values in between of lower_bound and upper_bound
// if lower_bound and upper_bound are omitted, then values will be generated in the interval [0;1]
Matrix random_matrix(2, 3, true, lower_bound, upper_bound);
```

There are also special square matrices, which can be created using static function-members of Matrix class:
```cpp
Matrix zeros = Matrix::zeros(5); // creates zero matrix of order 5
Matrix ones = Matrix::ones(5); // creates matrix of order 5 filled with ones
Matrix identity = Matrix::identity(5); // creates identity matrix of order 5
```

---

## Contributors

- Yurii Khomiak

---

## License & copyright

© Yurii Khomiak

Licensed under the [MIT License](LICENSE).
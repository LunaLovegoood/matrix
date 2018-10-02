// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#include "matrix.h"

#include <iostream>
#include <new>
#include <vector>
#include <random>


namespace {
    using MapFunc = double(*)(double);

    using random_engine = std::mt19937;
    using uniform_distribution = std::uniform_real_distribution<double>;
}


namespace matrix {

    //
    //
    // Constructors and destructors
    //
    //

    // Copy constructor
    // Throws: std::bad_alloc
    Matrix::Matrix(const Matrix &matrix) {
        try {
            copy(matrix);
        }
        catch (const std::bad_alloc &e) {
            throw e;
        }
    }

    // Move constructor
    Matrix::Matrix(Matrix &&matrix)
    : matrix_(matrix.matrix_), rows_(matrix.rows_), cols_(matrix.cols_), 
      details_(matrix.details_), det_(std::move(matrix.det_)) {

        matrix.set_default();
    }

    // Constructor
    // rows and cols represent respectively number of rows and columns in matrix
    // If random is set to true: matrix will be filled with random numbers, otherwise: with zeros
    // lower_bound and upper_bound set min and max values for possible random numbers
    // By default: random = false, lower_bound = 0, upper_bound = 1
    // Throws: std::bad_alloc,
    //         IncorrectBoundsForRandom
    Matrix::Matrix(int rows, int cols, bool random, long lower_bound, long upper_bound)
    : rows_(rows), cols_(cols) {

        details_.is_init_ = true;

        try {

            if (random) {
                allocate_memory(false);
                fill_random(lower_bound, upper_bound);
            }
            else {
                allocate_memory();
            }
        }
        catch (const std::bad_alloc &e) {
            throw e;
            return;
        }

        if (rows == cols)
            details_.is_square_ = true;
    }

    // Constructor
    // rows and cols represent respectively number of rows and columns in matrix
    // matrix is an 2d array which content is to be assigned to class member matrix_
    // Throws: std::bad_alloc
    Matrix::Matrix(int rows, int cols, double **matrix)
    : rows_(rows), cols_(cols) {

        details_.is_init_ = true;

        try {
            allocate_memory(false);

            for (int i = 0; i < rows_; i++)
                for (int j = 0; j < cols_; j++)
                    matrix_[i][j] = matrix[i][j];
        }
        catch (const std::bad_alloc &e) {
            throw e;
            return;
        }

        if (rows_ == cols_)
            details_.is_square_ = true;
    }

    // Constructs matrix from std::vector of std::vectors
    // Throws: std::bad_alloc
    Matrix::Matrix(std::vector< std::vector<double> > matrix) 
    : rows_(matrix.size()), cols_(matrix[0].size()) {

        details_.is_init_ = true;

        try {
            allocate_memory(false);

            for (int i = 0; i < rows_; i++)
                for (int j = 0; j < cols_; j++)
                    matrix_[i][j] = matrix[i][j];
        }
        catch (const std::bad_alloc &e) {
            throw e;
            return;
        }

        if (rows_ == cols_)
            details_.is_square_ = true;
    }

    // Construct special square matrix
    // Throws: std::bad_alloc
    Matrix::Matrix(int order, SpecType spec_type) 
    : rows_(order), cols_(order) {

        details_.is_init_ = true;
        details_.is_square_ = true;

        try {
            switch (spec_type) {
            case SpecType::ZEROS: // Creates zeros matrix
                allocate_memory();
                break;

            case SpecType::ONES: // Creates ones matrix
                allocate_memory(false);

                for (int i = 0; i < order; i++)
                    for (int j = 0; j < order; j++)
                        matrix_[i][j] = 1.0;
                break;

            case SpecType::IDENTITY: // Creates identity matrix
                allocate_memory();

                for (int i = 0; i < order; i++)
                    matrix_[i][i] = 1.0;
                break;
            }
        }
        catch (const std::bad_alloc &e) {
            throw e;
            return;
        }
    }

    // Destructor
    // Frees up memory allocated for matrix
    Matrix::~Matrix() {
        free_matrix();
    }

    //
    //
    // Overloaded operators
    //
    //

    // Copy assignment
    // Throws: std::bad_alloc
    Matrix& Matrix::operator=(const Matrix &matrix) {

        try {
            free_matrix();
            copy(matrix);
        }
        catch (const std::bad_alloc &e) {
            throw e;
        }

        return *this;
    }

    // Move assignment
    Matrix& Matrix::operator=(Matrix &&matrix) {

        if (&matrix == this) return *this;

        free_matrix();

        matrix_ = matrix.matrix_;
        rows_ = matrix.rows_;
        cols_ = matrix.cols_;
        details_ = matrix.details_;
        det_ = std::move(matrix.det_);

        matrix.set_default();

        return *this;
    }

    // Shorthand assignment for adding matrix
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix& Matrix::operator+=(const Matrix &matrix) {

        try {
            *this = ((*this) + matrix);
        }
        catch (...) {
            throw;
        }

        return *this;
    }

    // Shorthand assignment for adding number
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator+=(const double &arg) {

        try {
            *this = ((*this) + arg);
        }
        catch (UninitializedMatrix e) {
            throw e;
        }

        return *this;
    }

    // Shorthand assignment for subtracting Matrix
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix& Matrix::operator-=(const Matrix &matrix) {

        try {
            *this = ((*this) - matrix);
        }
        catch (...) {
            throw;
        }

        return *this;
    }

    // Shorthand assignment for subtracting number
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator-=(const double &arg) {

        try {
            *this = ((*this) - arg);
        }
        catch (UninitializedMatrix e) {
            throw e;
        }

        return *this;
    }

    // Shorthand assignment for multiplying by matrix
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix& Matrix::operator*=(const Matrix &matrix) {

        try {
            *this = ((*this) * matrix);
        }
        catch (...) {
            throw;
        }

        return *this;
    }

    // Shorthand assignment for multuplying by number
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator*=(const double &arg) {

        try {
            *this = ((*this) * arg);
        }
        catch (UninitializedMatrix e) {
            throw e;
        }

        return *this;
    }

    // Shorthand assignment for dividing by number
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator/=(const double &arg) {

        try {
            *this = ((*this) / arg);
        }
        catch (UninitializedMatrix e) {
            throw;
        }

        return *this;
    }

    // Prefix increment
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator++() {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return *this;
        }

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j]++;

        return *this;
    }

    // Postfix increment
    // Throws: UninitializedMatrix
    Matrix Matrix::operator++(int x) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix temp(*this);
        ++(*this);

        return temp;
    }

    // Prefix decrement
    // Throws: UninitializedMatrix
    Matrix& Matrix::operator--() {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return *this;
        }

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j]--;

        return *this;
    }

    // Postfix decrement
    // Throws: UninitializedMatrix
    Matrix Matrix::operator--(int x) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix temp(*this);
        --(*this);

        return temp;
    }

    // Returns sum matrix of calling and arg matrices
    // Throws: UninitializedMatrix,
    //           IncorrectDimensions
    Matrix Matrix::operator+(const Matrix &matrix) const {

        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!is_correct_dimensions(matrix)) {
            throw IncorrectDimensions();
            return {};
        }

        Matrix result(rows_, cols_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result.matrix_[i][j] = matrix_[i][j] + matrix.matrix_[i][j];

        return result;
    }

    // Returns difference between calling matrix and arg matrix
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix Matrix::operator-(const Matrix &matrix) const {

        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!is_correct_dimensions(matrix)) {
            throw IncorrectDimensions();
            return {};
        }

        Matrix result(rows_, cols_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result.matrix_[i][j] = matrix_[i][j] - matrix.matrix_[i][j];

        return result;
    }

    // Returns product of calling and arg matrices
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix Matrix::operator*(const Matrix &matrix) const {

        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!is_correct_mult_dimensions(matrix)) {
            throw IncorrectDimensions();
            return {};
        }

        Matrix result(rows_, matrix.cols_);
        if (rows_ == matrix.cols_)
            result.details_.is_square_ = true;

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                for (int k = 0; k < cols_; k++)
                    result.matrix_[i][j] += matrix_[i][k] * matrix.matrix_[k][j];

        return result;
    }

    // Returns matrix in which each element has an opposite sign
    // Throws: UninitializedMatrix
    Matrix Matrix::operator-() const {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(rows_, cols_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result.matrix_[i][j] = -matrix_[i][j];

        return result;
    }

    // Compares two matrices for equality
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    bool Matrix::operator==(const Matrix &matrix) const {

        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!is_correct_mult_dimensions(matrix)) {
            throw IncorrectDimensions();
            return {};
        }

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                if (!compare_doubles(matrix_[i][j], matrix.matrix_[i][j]))
                    return false;

        return true;
    }

    //
    //
    // Friend functions
    //
    //

    // Returns sum matrix of calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator+(const double &number, const Matrix &matrix) {

        if (!matrix.is_init()) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(matrix.rows_, matrix.cols_);

        for (int i = 0; i < matrix.rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                result.matrix_[i][j] = matrix.matrix_[i][j] + number;

        return result;
    }

    // Returns sum matrix of calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator+(const Matrix &matrix, const double &number) {
        try {
            return (number + matrix);
        }
        catch (UninitializedMatrix e) {
            throw;
            return {};
        }
    }

    // Returns difference between calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator-(const double &number, const Matrix &matrix) {
        try {
            return ((-matrix) + number);
        }
        catch (UninitializedMatrix e) {
            throw;
            return {};
        }
    }

    // Returns difference between calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator-(const Matrix &matrix, const double &number) {

        if (!matrix.is_init()) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(matrix.rows_, matrix.cols_);

        for (int i = 0; i < matrix.rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                result.matrix_[i][j] = matrix.matrix_[i][j] - number;

        return result;
    }

    // Returns product of calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator*(const double &number, const Matrix &matrix) {

        if (!matrix.is_init()) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(matrix.rows_, matrix.cols_);

        for (int i = 0; i < matrix.rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                result.matrix_[i][j] = matrix.matrix_[i][j] * number;

        return result;
    }

    // Returns product of calling matrix and number
    // Throws: UninitializedMatrix
    Matrix operator*(const Matrix &matrix, const double &number) {
        try {
            return (number * matrix);
        }
        catch (UninitializedMatrix e) {
            throw;
            return {};
        }
    }

    // // Returns matrix with every element divided by number
    // Throws: UninitializedMatrix
    Matrix operator/(const Matrix &matrix, const double &number) {

        if (!matrix.is_init()) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(matrix.rows_, matrix.cols_);

        for (int i = 0; i < matrix.rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                result.matrix_[i][j] = matrix.matrix_[i][j] / number;

        return result;
    }

    // Outputs matrix with previously set or default precision
    // Throws: UninitializedMatrix
    std::ostream& operator<<(std::ostream &stream, const Matrix &matrix) {

        if (!matrix.is_init()) {
            throw UninitializedMatrix();
            return stream;
        }

        std::streamsize old_precision = stream.precision(matrix.precision_);

        stream << matrix.rows_ << " " << matrix.cols_ << " " << matrix.details_ << "\n";

        for (int i = 0; i < matrix.rows_; i++) {
            for (int j = 0; j < matrix.cols_; j++)
                stream << matrix.matrix_[i][j] << " ";
            stream << "\n";
        }

        stream << std::endl;
        stream.precision(old_precision);

        return stream;
    }

    // Reads matrix
    // Throws: std::bad_alloc
    std::istream& operator>>(std::istream &stream, Matrix &matrix) {

        matrix.free_matrix();

        stream >> matrix.rows_ >> matrix.cols_ >> matrix.details_;

        try {
            matrix.allocate_memory();
        }
        catch (const std::bad_alloc &e) {
            throw e;
            return stream;
        }

        for (int i = 0; i < matrix.rows_; i++)
            for (int j = 0; j < matrix.cols_; j++)
                stream >> matrix.matrix_[i][j];

        return stream;
    }

    //
    //
    // Additional functions
    //
    //

    // Returns Hadamard product of calling matrix and arg
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix Matrix::hadm_product(const Matrix &matrix) const {

        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!is_correct_dimensions(matrix)) {
            throw IncorrectDimensions();
            return {};
        }

        Matrix result(rows_, cols_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result.matrix_[i][j] = matrix_[i][j] * matrix.matrix_[i][j];

        return result;
    }

    // Returns transposed matrix of calling one
    // Throws: UninitializedMatrix
    Matrix Matrix::transpose() const {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }

        Matrix result(cols_, rows_);

        for (int i = 0; i < cols_; i++)
            for (int j = 0; j < rows_; j++)
                result.matrix_[i][j] = matrix_[j][i];

        return result;
    }

    // Returns determinant of the matrix
    // Throws: UninitializedMatrix,
    //         DetDoesNotExist
    double Matrix::det() const {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }
        else if (!details_.is_square_) {
            throw DetDoesNotExist();
            return {};
        }

        if (!det_) {
            det_.reset(new double(calculate_det()));
        }

        return (*det_);
    }

    // Calculates determinant of the matrix
    double Matrix::calculate_det() const noexcept {
        
        double result = 1.0;

        if (details_.is_triangular_) {
            for (int i = 0; i < rows_; i++)
                result *= matrix_[i][i];
            return result;
        }
        
        // TODO: implement common determinant
        return 0.0;
    }

    // Returns merged matrix
    // Throws: UninitializedMatrix,
    //         IncorrectDimensions
    Matrix Matrix::merge(const Matrix &matrix) const {
        
        if (!are_initialized(matrix)) {
            throw UninitializedMatrix();
            return {};
        }
        else if (rows_ != matrix.rows_) {
            throw IncorrectDimensions();
            return {};
        }

        Matrix result(rows_, cols_ + matrix.cols_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                result.matrix_[i][j] = matrix_[i][j];

        for (int i = 0; i < rows_; i++)
            for (int j = cols_, k = 0; j < result.cols_; j++, k++)
                result.matrix_[i][j] = matrix.matrix_[i][k];

        return result;
    }

    // Applies given function to each element of calling matrix
    // Throws: UninitializedMatrix
    void Matrix::map(MapFunc map_function) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return;
        }

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = map_function(matrix_[i][j]);
    }

    // Creates square matrix of specified order filled with zeros
    // Throws: std::bad_alloc
    Matrix Matrix::zeros(int order) {
        try {
            return { order, SpecType::ZEROS };
        }
        catch (std::bad_alloc &e) {
            throw e;
            return {};
        }
    }

    // Creates square matrix of specified order filled with ones
    // Throws: std::bad_alloc
    Matrix Matrix::ones(int order) {
        try {
            return { order, SpecType::ONES };
        }
        catch (std::bad_alloc &e) {
            throw e;
            return {};
        }
    }

    // Creates identity matrix of specified order
    // Throws: std::bad_alloc
    Matrix Matrix::identity(int order) {
        try {
            return { order, SpecType::IDENTITY };
        }
        catch (std::bad_alloc &e) {
            throw e;
            return {};
        }
    }

    // Returns matrix in form of std::vector of std::vectors
    // Throws: UninitializedMatrix
    std::vector< std::vector<double> > Matrix::get_matrix() const {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return {};
        }

        std::vector< std::vector<double> > matrix(rows_);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix[i].push_back(matrix_[i][j]);
        
        return matrix;
    }

    // Fills matrix with given arg number
    // Throws: UninitializedMatrix
    void Matrix::fill(const double &arg) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return;
        }

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = arg;
    }

    // Fills matrix with random values in the interval between lower_bound and upper_bound
    // Throws: UninitializedMatrix,
    //         IncorrectBoundsForRandom
    void Matrix::fill_random(long lower_bound, long upper_bound) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return;
        }
        else if (lower_bound >= upper_bound) {
            throw IncorrectBoundsForRandom();
            return;
        }

        random_engine engine;
        uniform_distribution distribution(lower_bound, upper_bound);

        for (int i = 0; i < rows_; i++)
            for (int j = 0; j < cols_; j++)
                matrix_[i][j] = distribution(engine);
    }

    // Sets number on the position defined by row and col arguments
    // Throws: UninitializedMatrix,
    //         MatrixOutOfBounds
    void Matrix::set_element(int row, int col, const double &number) {

        if (!details_.is_init_) {
            throw UninitializedMatrix();
            return;
        }
        else if (row < 0 || row >= rows_ ||
            col < 0 || col >= cols_) {
            throw MatrixOutOfBounds();
            return;
        }

        matrix_[row][col] = number;
    }

    // Sets default values to matrix members
    void Matrix::set_default() noexcept {

        matrix_ = nullptr;
        details_ = {};
        rows_ = 0;
        cols_ = 0;
        det_.reset();
    }

    // Frees up allocated for matrix_ space
    void Matrix::free_matrix() noexcept {

        if (details_.is_init_)
        {
            for (int i = 0; i < rows_; i++)
                delete[] matrix_[i];
            delete[] matrix_;

            set_default();
        }
    }

    // Copies given matrix arg into invoking one
    // Throws: std::bad_alloc
    void Matrix::copy(const Matrix &matrix) {

        rows_ = matrix.rows_;
        cols_ = matrix.cols_;
        details_ = matrix.details_;
        det_.reset(new double(*(matrix.det_)));

        try {
            allocate_memory(false);

            for (int i = 0; i < matrix.rows_; i++)
                for (int j = 0; j < matrix.cols_; j++)
                    matrix_[i][j] = matrix.matrix_[i][j];
        }
        catch (const std::bad_alloc &e) {
            matrix_ = nullptr;
            throw e;
        }
    }

    // Allocates memory for matrix
    // Throws: std::bad_alloc
    void Matrix::allocate_memory(bool init) {

        try {
            matrix_ = new double *[rows_];

            if (init) {
                for (int i = 0; i < rows_; i++)
                    matrix_[i] = new double[cols_]();
            }
            else {
                for (int i = 0; i < rows_; i++)
                    matrix_[i] = new double[cols_];
            }
        }
        catch (const std::bad_alloc &e) {
            matrix_ = nullptr;
            details_ = {};
            throw e;
        }
    }

}

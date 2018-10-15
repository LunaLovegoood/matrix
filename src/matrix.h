// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#ifndef MATRIX_H_
#define MATRIX_H_

#include "matrix_detail.h"
#include "matrix_exception.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>


namespace {
    using MapFunc = double(*)(double);
}


namespace matrix {

    // Matrix provides basic functions to handle matrix operations
    // TODO: rework Matrix to be template
    class Matrix {

    public:
        Matrix() {}
        Matrix(const Matrix &matrix);
        Matrix(Matrix &&matrix);
        Matrix(int rows, int cols, bool random = false, long lower_bound = 0, long upper_bound = 1);
        Matrix(int rows, int cols, double **matrix);
        Matrix(const std::vector< std::vector<double> > &matrix);
        ~Matrix();
        
        Matrix hadm_product(const Matrix &matrix) const;
        Matrix transpose() const;
        Matrix row_echelon() const;
        double det() const;
        void map(MapFunc map_function);

        Matrix merge(const Matrix &matrix) const;
        void fill(const double &arg);
        void fill_random(long lower_bound = 0, long upper_bound = 1);
        void set_element(int row, int col, const double &number);
        void copy(const Matrix &matrix);

        // Creating special matrices
        static Matrix zeros(int order);
        static Matrix ones(int order);
        static Matrix identity(int order);

        std::vector< std::vector<double> > get_matrix() const;
        int rows() const noexcept { return rows_; }
        int cols() const noexcept { return cols_; }
        int precision() const noexcept { return precision_; }
        bool is_init() const noexcept { return details_.is_init_; }

        // Sets matrix precision for output to given precision if it is greater than 0 and less or equal than 25,
        // otherwise precision remains unchanged
        void set_precision(int precision) noexcept {
            if (precision > 0 && precision <= 25)
                precision_ = precision;
        }

        Matrix& operator=(const Matrix &matrix);
        Matrix& operator=(Matrix &&matrix);
        Matrix& operator+=(const Matrix &matrix);
        Matrix& operator+=(const double &arg);
        Matrix& operator-=(const Matrix &matrix);
        Matrix& operator-=(const double &arg);
        Matrix& operator*=(const Matrix &matrix);
        Matrix& operator*=(const double &arg);
        Matrix& operator/=(const Matrix &matrix) = delete;
        Matrix& operator/=(const double &arg);
        Matrix& operator++();
        Matrix operator++(int x);
        Matrix& operator--();
        Matrix operator--(int x);
        Matrix operator+(const Matrix &matrix) const;
        Matrix operator-(const Matrix &matrix) const;
        Matrix operator*(const Matrix &matrix) const;
        Matrix operator/(const Matrix &matrix) const = delete;
        Matrix operator-() const;

        // Comparison operators
        friend bool operator==(const Matrix &first, const Matrix &second);
        friend bool operator!=(const Matrix &first, const Matrix &second);
        friend bool operator<(const Matrix &first, const Matrix &second) = delete;
        friend bool operator>(const Matrix &first, const Matrix &second) = delete;
        friend bool operator<=(const Matrix &first, const Matrix &second) = delete;
        friend bool operator>=(const Matrix &first, const Matrix &second) = delete;

        friend Matrix operator+(const double &number, const Matrix &matrix);
        friend Matrix operator+(const Matrix &matrix, const double &number);
        friend Matrix operator-(const double &number, const Matrix &matrix);
        friend Matrix operator-(const Matrix &matrix, const double &number);
        friend Matrix operator*(const double &number, const Matrix &matrix);
        friend Matrix operator*(const Matrix &matrix, const double &number);
        friend Matrix operator/(const double &number, const Matrix &matrix) = delete;
        friend Matrix operator/(const Matrix &matrix, const double &number);

        friend std::ostream& operator<<(std::ostream &stream, const Matrix &matrix);
        friend std::istream& operator>>(std::istream &stream, Matrix &matrix);

    private:
        double **matrix_{ nullptr };
        int rows_{ 0 };
        int cols_{ 0 };

        MatrixDetail details_{};
        int precision_{ 8 };

        mutable std::unique_ptr<double> det_{ nullptr };

        enum class SpecType { ZEROS, ONES, IDENTITY }; // Options for special matrix types
        Matrix(int order, SpecType spec_type);       // Constructor, which creates special matrices

        // Returns true if given matrix is appropriate for addition,subtraction, comparison or merging
        bool is_correct_dimensions(const Matrix &arg) const noexcept {
            return ((rows_ == arg.rows_) && (cols_ == arg.cols_));
        }
        // Returns true if given matrix is appropriate for multiplication
        bool is_correct_mult_dimensions(const Matrix &arg) const noexcept {
            return (cols_ == arg.rows_);
        }
        // Returns true if calling and given matrix are initialized
        bool are_initialized(const Matrix &arg) const noexcept {
            return ((details_.is_init_) && (arg.is_init()));
        }
        // Compares two doubles 
        bool compare_doubles(double first, double second) const noexcept {
            return (std::abs(first - second) < 0.000002) ? true : false;
        }

        double calculate_det() const noexcept;

        int forward_eliminate() noexcept;
        void back_substitution() noexcept;
        void swap_rows(int first, int second) noexcept;
        int max_in_column(int row, int col) noexcept;

        void set_default() noexcept;

        void free_matrix() noexcept;
        void allocate_memory(bool init = true);
    };

}


#endif // MATRIX_H_

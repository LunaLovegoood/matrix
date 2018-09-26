// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#ifndef MATRIX_H_
#define MATRIX_H_

#include "matrix_detail.h"
#include "matrix_exception.h"

#include <iostream>


namespace {
	using MapFunc = double(*)(double);
}


namespace matrix {

	// Matrix provides basic functions to handle matrix operations
	class Matrix {

	public:
		Matrix() = default;
		Matrix(const Matrix &matrix);
		Matrix(Matrix &&matrix);
		Matrix(int rows, int cols, bool random = false, long lower_bound = 0, long upper_bound = 1);
		Matrix(int rows, int cols, double **matrix);
		~Matrix();

		Matrix hadm_product(const Matrix &matrix) const;
		Matrix transpose() const;
		Matrix merge(const Matrix &matrix) const;
		void map(MapFunc map_function);
		void fill(const double &arg);
		void fill_random(long lower_bound = 0, long upper_bound = 1);
		void set_element(int row, int col, const double &number);
		void copy(const Matrix &matrix);

		int get_rows() const { return rows_; }
		int get_cols() const { return cols_; }
		int get_precision() const { return precision_; }

		// Sets matrix precision for output to given precision if it is greater than 0 and less or equal than 25,
		// otherwise precision remains unchanged
		void set_precision(int precision) {
			if (precision > 0 && precision <= 25)
				precision_ = precision;
		}

		bool is_init() const { return details_.is_init_; }

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
		int precision_{ 10 };

		double det_{ 0 };

		// Returns true if given matrix is appropriate for addition or subtraction
		bool is_correct_dimensions(const Matrix &arg) const {
			return ((rows_ == arg.rows_) && (cols_ == arg.cols_));
		}
		// Returns true if given matrix is appropriate for multiplication
		bool is_correct_mult_dimensions(const Matrix &arg) const {
			return (cols_ == arg.rows_);
		}
		// Returns true if calling and given matrix are initialized
		bool are_initialized(const Matrix &arg) const {
			return ((details_.is_init_) && (arg.is_init()));
		}

		void set_default();

		void free_matrix();
		void allocate_memory();
	};

}


#endif // MATRIX_H_

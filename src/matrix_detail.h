// Matrix
// Copyright (C) 2018 Yurii Khomiak 
// Yurii Khomiak licenses this file to you under the MIT license. 
// See the LICENSE file in the project root for more information.

#ifndef MATRIX_DETAIL_H_
#define MATRIX_DETAIL_H_

#include <iostream>


namespace matrix {

	// MatrixDetail contains additional information about matrix
	struct MatrixDetail {
		bool is_init_{ false };
		bool is_square_{ false };
		bool is_triangular_{ false };

		MatrixDetail() = default;
		~MatrixDetail() = default;

		MatrixDetail(const MatrixDetail &arg) = default;
		MatrixDetail(MatrixDetail &&arg) = default;

		MatrixDetail& operator=(const MatrixDetail &arg) = default;
		MatrixDetail& operator=(MatrixDetail &&arg) = default;

		friend std::ostream& operator<<(std::ostream &stream, const MatrixDetail &details);
		friend std::istream& operator>>(std::istream &stream, MatrixDetail &details);
	};

}


#endif // MATRIX_DETAIL_H_

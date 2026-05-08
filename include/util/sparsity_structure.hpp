#pragma once

#include <Eigen/SparseCore>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>


namespace GV
{
	struct SparseMatImage
	{
		const Eigen::SparseMatrix<double,Eigen::RowMajor>& mat;
		SparseMatImage(const Eigen::SparseMatrix<double,Eigen::RowMajor>& matrix) : mat(matrix) {}

		void save_as(const std::string& filename) const
		{
			const int rows = static_cast<int>(mat.rows());
			const int cols = static_cast<int>(mat.cols());

			if (rows==0 or cols==0) {
				throw std::runtime_error("SparseMatImage: matrix has zero dimension(s).");
			}

			//24-bit (3-byte) RGB value per pixel
			const int rowBytes  = cols * 3;
			const int paddedRow = (rowBytes+3) & ~3; //round up to the nearest multiple of 4
			const int pixelDataSz = paddedRow * rows;

			//file header
			constexpr int headerSz = 54;
			const int fileSz = headerSz + pixelDataSz;

			//build the pixel buffer
			std::vector<uint8_t> pixels(pixelDataSz, 0xFF); //all white

			//mark every explicitly stored entry
			//gray if the stored entry is zero
			//black if the stored intry is non-zero
			auto setPixel = [&](int r, int c, uint8_t R, uint8_t G, uint8_t B)
			{
				//r,c is the matrix coordinate
				//only call this method if mat(r,c) is explicitly stored
				int bmpRow = (rows - 1 - r);
				int idx    = bmpRow * paddedRow + c*3;
				pixels[idx  ] = R;
				pixels[idx+1] = G;
				pixels[idx+2] = B;
			};

			//loop through the stored entries
			for (int r=0; r<rows; ++r) {
				for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(mat,r); it; ++it) {
					if (it.value() == 0.0) {
						setPixel(r, static_cast<int>(it.col()), 0x50, 0x50, 0x50);
					}
					else {
						setPixel(r, static_cast<int>(it.col()), 0x00, 0x00, 0x00);
					}
				}
			}

			//write file
			std::ofstream ofs(filename, std::ios::binary);
			if (!ofs) {
				throw std::runtime_error("SparseMatImage: cannot open file: " + filename);
			}

			auto writeU16 = [&](uint16_t v) {ofs.write(reinterpret_cast<const char*>(&v), 2);};
			auto writeU32 = [&](uint32_t v) {ofs.write(reinterpret_cast<const char*>(&v), 4);};
			auto writeI32 = [&](int32_t  v) {ofs.write(reinterpret_cast<const char*>(&v), 4);};

			//header
			ofs.write("BM", 2);
			writeU32(fileSz);
			writeU16(0);
			writeU16(0);
			writeU32(headerSz);

			writeU32(40); //header size
			writeI32(cols); //width
			writeI32(rows); //height
			writeU16(1);
			writeU16(24);
			writeU32(0);
			writeU32(pixelDataSz);
			constexpr int DPM = static_cast<int>(300.0/0.0254); //convert desired DPI to DPM
			writeI32(DPM);
			writeI32(DPM);
			writeU32(0);
			writeU32(0);

			ofs.write(reinterpret_cast<const char*>(pixels.data()), pixelDataSz);

			if (!ofs) {
				throw std::runtime_error("SparseMatImage: write error on file: " + filename);
			}
		}

		void save_as_bw(const std::string& filename) const
		{
			const int rows = static_cast<int>(mat.rows());
			const int cols = static_cast<int>(mat.cols());

			if (rows==0 or cols==0) {
				throw std::runtime_error("SparseMatImage: matrix has zero dimension(s).");
			}

			//1-bit per pixel, pack 8 pixels per byte
			const int paddedRow = ((cols+31) & ~31) / 8; //round up to the nearest multiple of 32 bits and convert to bytes
			const int pixelDataSz = paddedRow * rows;

			//file header
			constexpr int paletteSz = 8;
			constexpr int headerSz  = 54 + paletteSz;
			const     int fileSz    = headerSz + pixelDataSz;

			//build the pixel buffer
			std::vector<uint8_t> pixels(pixelDataSz, 0xFF); //all white

			//mark every explicitly stored entry
			//gray if the stored entry is zero
			//black if the stored intry is non-zero
			auto setPixel = [&](int r, int c)
			{
				//r,c is the matrix coordinate
				//only call this method if mat(r,c) is explicitly stored
				//because 8 pixels use a single byte, we need to do some
				//bit manipulation
				int bmpRow    = rows - 1 - r;
				int byteIdx   = bmpRow * paddedRow + c/8;
				int bitIdx    = 7 - (c%8);
				pixels[byteIdx] &= ~(1 << bitIdx);
			};

			//loop through the stored entries
			for (int r=0; r<rows; ++r) {
				for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(mat,r); it; ++it) {
					setPixel(r, static_cast<int>(it.col()));
				}
			}

			//write file
			std::ofstream ofs(filename, std::ios::binary);
			if (!ofs) {
				throw std::runtime_error("SparseMatImage: cannot open file: " + filename);
			}

			auto writeU16 = [&](uint16_t v) {ofs.write(reinterpret_cast<const char*>(&v), 2);};
			auto writeU32 = [&](uint32_t v) {ofs.write(reinterpret_cast<const char*>(&v), 4);};
			auto writeI32 = [&](int32_t  v) {ofs.write(reinterpret_cast<const char*>(&v), 4);};

			//header
			ofs.write("BM", 2);
			writeU32(fileSz);
			writeU16(0);
			writeU16(0);
			writeU32(headerSz);

			writeU32(40); //header size
			writeI32(cols); //width
			writeI32(rows); //height
			writeU16(1);
			writeU16(1);
			writeU32(0);
			writeU32(pixelDataSz);
			constexpr int DPM = static_cast<int>(300.0/0.0254); //convert desired DPI to DPM
			writeI32(DPM);
			writeI32(DPM);
			writeU32(2); //numbers in color palette
			writeU32(0);

			//color palette
			writeU32(0x0); //black
			writeU32(0x00FFFFFF); //white

			//pixel data
			ofs.write(reinterpret_cast<const char*>(pixels.data()), pixelDataSz);

			if (!ofs) {
				throw std::runtime_error("SparseMatImage: write error on file: " + filename);
			}
		}
	};
}
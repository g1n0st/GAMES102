#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

class Interpolation {
public:
	// input:
	static constexpr int POINT_LIMIT = 30; // Maximum size of input point set
	int n; // Number of input points
	float x[POINT_LIMIT], y[POINT_LIMIT];

public:
	// output:
	static constexpr float LEFT_LIMIT = 0.f; // Left end of sampling interval
	static constexpr float RIGHT_LIMIT = 10.f; // Right end of sampling interval
	static constexpr float SAMPLE_RATE = 0.05f; // Distance of adjoining sample points
	static constexpr int SAMPLE_POINTS = static_cast<int>((RIGHT_LIMIT - LEFT_LIMIT) / SAMPLE_RATE) + 1; // Number of total sample points
	float out_x[SAMPLE_POINTS], out_y[SAMPLE_POINTS];

	Interpolation(int num, const float *ix, const float *iy) {
		n = num;
		memcpy_s(x, num * sizeof(float), ix, num * sizeof(float));
		memcpy_s(y, num * sizeof(float), iy, num * sizeof(float));
		memset(out_y, 0, sizeof(float) * SAMPLE_POINTS);
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			out_x[i] = (LEFT_LIMIT + i * SAMPLE_RATE);
		}
	}

	void outputToCSV(const char *filename, std::string title = "") { // output results to CSV
		std::ofstream csv(filename, std::ios::out | std::ios::app);
		csv << title << ",";
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			csv << out_y[i] << char(i == SAMPLE_POINTS - 1 ? '\n' : ',');
		}

		csv.close();
	}

	virtual void evaluate() = 0; // interpolate
};

class LinearInterpolation : public Interpolation {
public:
	LinearInterpolation(int num, const float *ix, const float *iy) : Interpolation(num, ix, iy) {}

	void evaluate() final {
		int k = -1;
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			while (k + 1 < n && x[k + 1] < out_x[i]) ++k;
			if (k == -1 || out_x[i] > x[n - 1]) {
				out_y[i] = 0.0f;
			}
			else {
				float fact = (out_x[i] - x[k]) / (x[k + 1] - x[k]);
				out_y[i] = fact * y[k + 1] + (1.0f - fact) * y[k];
			}
		}
	}
};

class LagrangeInterpolation : public Interpolation {
public:
	LagrangeInterpolation(int num, const float *ix, const float *iy) : Interpolation(num, ix, iy) {}

	void evaluate() final {
		for (int h = 0; h < SAMPLE_POINTS; ++h) {
			float res = 0.0f;
			for (int i = 0; i < n; ++i) {
				float tmp = y[i];
				for (int j = 0; j < n; ++j) {
					if (i != j) {
						tmp = tmp * (out_x[h] - x[j]);
						tmp = tmp / (x[i] - x[j]);
					}
				}

				res += tmp;
			}
			out_y[h] = res;
		}
	}
};

class GaussInterpolation : public Interpolation {
public:
	float b0, sigma;
	GaussInterpolation(int num, const float *ix, const float *iy, 
		float _b0, float _sigma = 1.0f) : b0(_b0), sigma(_sigma), Interpolation(num, ix, iy) {}

	float gauss(float x, float x0) {
		return std::exp(-std::pow((x - x0) / sigma, 2.0f) / 2.0f);
	}
	void evaluate() final {
		MatrixXf A(n, n), B(n, 1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A(i, j) = gauss(x[i], x[j]);
			}
			B(i, 0) = y[i] - b0;
		}
		MatrixXf W = A.inverse() * B;
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			float res = b0;
			for (int j = 0; j < n; ++j) {
				res += W(j, 0) * gauss(out_x[i], x[j]);
			}
			out_y[i] = res;
		}
	}
};

class LeastSquareInterpolation : public Interpolation {
public:
	int m;
	LeastSquareInterpolation(int num, const float *ix, const float *iy, int _m) 
		: m(_m), Interpolation(num, ix, iy) {}

	void evaluate() final {
		MatrixXf A(n, m + 1);
		for (int i = 0; i < n; ++i) {
			float pow = 1.0f;
			for (int j = m; j >= 0; --j) {
				A(i, j) = pow;
				pow *= x[i];
			}
		}
		
		MatrixXf B(n, 1);
		for (int i = 0; i < n; ++i) {
			B(i, 0) = y[i];
		}

		MatrixXf W = (A.transpose() * A).inverse() * A.transpose() * B;
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			float pow = 1.0f, res = 0.0f;
			for (int j = m; j >= 0; --j) {
				res += pow * W(j, 0);
				pow *= out_x[i];
			}
			out_y[i] = res;
		}
	}
};

class RidgeRegressionInterpolation : public Interpolation {
public:
	int m;
	float lambda;
	RidgeRegressionInterpolation(int num, const float *ix, const float *iy, 
		int _m, float _lambda) : m(_m), lambda(_lambda), Interpolation(num, ix, iy) {}

	void evaluate() final {
		MatrixXf A(n, m + 1);
		for (int i = 0; i < n; ++i) {
			float pow = 1.0f;
			for (int j = m; j >= 0; --j) {
				A(i, j) = pow;
				pow *= x[i];
			}
		}

		MatrixXf B(n, 1);
		for (int i = 0; i < n; ++i) {
			B(i, 0) = y[i];
		}

		MatrixXf W = (A.transpose() * A + lambda * MatrixXf::Identity(m + 1, m + 1)).inverse() * A.transpose() * B;
		for (int i = 0; i <= m; i++) std::cout << W(i, 0) << " ";
		for (int i = 0; i < SAMPLE_POINTS; ++i) {
			float pow = 1.0f, res = 0.0f;
			for (int j = m; j >= 0; --j) {
				res += pow * W(j, 0);
				pow *= out_x[i];
			}
			out_y[i] = res;
		}
	}
};


int main() {
	std::ifstream dataflow("data.txt", std::ios::in);
	int num;
	float x[Interpolation::POINT_LIMIT], y[Interpolation::POINT_LIMIT];
	dataflow >> num;
	for (int i = 0; i < num; ++i) {
		dataflow >> x[i];
	}
	for (int i = 0; i < num; ++i) {
		dataflow >> y[i];
	}

	dataflow.close();

	LinearInterpolation inter0(num, x, y);
	LagrangeInterpolation inter1(num, x, y);
	GaussInterpolation inter2(num, x, y, 0.0f);
	LeastSquareInterpolation inter3(num, x, y, 2);
	RidgeRegressionInterpolation inter4(num, x, y, 5, 10000.0f);

	inter0.evaluate();
	inter1.evaluate();
	inter2.evaluate();
	inter3.evaluate();
	inter4.evaluate();

	inter0.outputToCSV("result.csv", "LinearInterpolation");
	inter1.outputToCSV("result.csv", "LagrangeInterpolation");
	inter2.outputToCSV("result.csv", "GaussInterpolation");
	inter3.outputToCSV("result.csv", "LeastSquareInterpolation");
	inter4.outputToCSV("result.csv", "RidgeRegressionInterpolation");
	return 0;
}
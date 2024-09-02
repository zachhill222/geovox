#pragma once


namespace GeoVox::utils{
	class Point{
	public:
		//////////////////////////////////////////////////////////////////
		////////////////// CONSTRUCTOR AND DESTRUCTOR ////////////////////
		//////////////////////////////////////////////////////////////////
		Point(): _len(3) {
			_data = new double[_len];
		}

		Point(const int _len): _len(_len){
			_data = new double[_len];
		}

		// ~Point(){
		// 	delete[] _data;
		// }

		
		//////////////////////////////////////////////////////////////////
		/////////////////// DATA ACCESS METHODS //////////////////////////
		//////////////////////////////////////////////////////////////////
		//access and set data
		double& operator[](int idx);
		double  operator[](int idx) const;


		//////////////////////////////////////////////////////////////////
		////////////////// ARITHMETIC METHODS ////////////////////////////
		//////////////////////////////////////////////////////////////////
		//additions
		Point* operator+=(const Point& other);
		Point  operator+(const Point& other);

		//subtractions
		Point* operator-=(const Point& other);
		Point  operator-(const Point& other);

		//element-wise multiplication
		Point* operator*=(const Point& other);
		Point  operator*(const Point& other);

		//element-wise division
		Point* operator/=(const Point& other);
		Point  operator/(const Point& other);

		//scalar multiplication
		Point* operator*=(const double& scalar);
		Point  operator*(const double& scalar);
		Point* operator/=(const double& scalar);
		Point  operator/(const double& scalar);

		
		//////////////////////////////////////////////////////////////////
		//////////////// COMPARISION METHODS /////////////////////////////
		//////////////////////////////////////////////////////////////////
		bool operator<(const Point& other);
		bool operator<=(const Point& other);
		bool operator>(const Point& other);
		bool operator>=(const Point& other);
		bool operator==(const Point& other);
		bool operator!=(const Point& other);


		//////////////////////////////////////////////////////////////////
		/////////////// CONVENIENCE METHODS //////////////////////////////
		//////////////////////////////////////////////////////////////////
		int len(); //get length
		void fill(const double& scalar); //set all elements to scalar
		double max(); //get maximum element
		int maxind(); //get lowest index of maximum element
		double min(); //get minimum element
		int minind(); //get lowest index of minimum element
		double sum(); //sum all elements
		Point normalize(); //normalize NOT in-place to a Euclidean norm of 1


		//////////////////////////////////////////////////////////////////
		////////////////// DISTANCE METHODS //////////////////////////////
		//////////////////////////////////////////////////////////////////
		double norm2();//Euclidean norm squared
		double norm();//Euclidean norm
		double infnorm();//max-norm
		double dist2(const Point& other); //Euclidean distance squared
		double dist(const Point& other); //Euclidean distance
		double dist_inf(const Point& other); //max-norm distance


		//////////////////////////////////////////////////////////////////
		////////////////// LINEAR ALGEBRA METHODS ////////////////////////
		//////////////////////////////////////////////////////////////////
		double dot(const Point& other); //dot product
		Point cross(const Point& other); //cross product, only if _len==3, but no error check


	// private:
		double* _data;
		int _len;

	};
}
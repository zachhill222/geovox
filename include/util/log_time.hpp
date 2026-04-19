#pragma once

#include <chrono>
#include <mutex>
#include <iostream>
#include <string_view>
#include <thread>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//A log class to print to an output stream in a thread safe manner
	//It also tracks the time elapsed since the beginning of the program and which thread called it
	struct Logger
	{
		//starting time of the program, sychronization mutex, and output stream
		static inline std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
		static inline std::mutex mtx;
		static inline std::ostream* os = &std::cout;

		//change the output stream
		static void set_output(std::ostream& os_) {os = &os_;}

		//write to the ouput stream (thread safe)
		static void log(std::string_view msg) {
			const auto now = std::chrono::steady_clock::now();
			const double elapsed = std::chrono::duration<double>(now - start_time).count();

			std::lock_guard<std::mutex> lock(mtx);

			#ifdef _OPENMP
			*os   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					<< "omp_thread=" << omp_get_thread_num() << "] "
					<< msg << "\n";
			#else
			*os   << "[t=" << std::fixed << std::setprecision(4) << elapsed << "s | "
					<< "thread=" << std::this_thread::get_id() << "] "
					<< msg << "\n";
			#endif

			os -> flush();
		}
	};


	//A logging method built to time the duration of some subroutine
	//Initialize it with a label and it prints on 
	struct LogTime
	{
		std::string_view label;
		std::chrono::steady_clock::time_point mark_start;

		explicit LogTime(std::string_view label) : label{label}, mark_start{std::chrono::steady_clock::now()} {}

		~LogTime() {
			const auto now = std::chrono::steady_clock::now();
			const double elapsed = std::chrono::duration<double>(now - mark_start).count();
			Logger::log(std::string(label) + " : " + std::to_string(elapsed) + "s");
		}
	};


}


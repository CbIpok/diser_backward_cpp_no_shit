#define NOMINMAX 
#include "managers.h"
#include <windows.h>
#include <string>
#include <sstream>
#include <netcdf.h>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <future>
#include <regex>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;
#define GET_TIME(code) do { code; } while(0)

int open_nc_file(const std::string& filename, int& ncid) {
    int retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "Error opening file " << filename << " : "
            << nc_strerror(retval) << std::endl;
    }
    return retval;
}

std::vector<double> read_nc_file(const std::string& filename, int y_start, int y_end) {
    std::vector<double> result;
    fs::path file(filename);
    std::cout << "Loading file (mapped): " << file << std::endl;

    // Dimensions in the .nc
    const size_t T = 2001;
    const size_t Y = 512;
    const size_t X = 512;

    int local_y_end = y_end;
    if (static_cast<size_t>(local_y_end) > Y)
        local_y_end = static_cast<int>(Y);
    int region_height = local_y_end - y_start;
    if (region_height <= 0)
        return result;

    size_t totalElements = T * region_height * X;
    size_t totalBytes = totalElements * sizeof(double);

    // Create a unique shared-memory name
    std::ostringstream shmNameStream;
    shmNameStream << "Local\\MySharedMemory_"
        << GetCurrentProcessId() << "_"
        << GetTickCount()
        << file.extension().string();
    std::string shmName = shmNameStream.str();

    // Create the file‐mapping
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        static_cast<DWORD>(totalBytes),
        shmName.c_str()
    );
    if (!hMapFile) {
        std::cerr << "Could not create file mapping object ("
            << GetLastError() << ")\n";
        return result;
    }

    fs::path exePath = "D:/dmitrienkomy/cpp/get_coofs/x64/Debug";  // typically the bin\Debug folder
    std::string childExe = (exePath / "nc_reader.exe").string();
    std::ostringstream oss;
    oss << "\"" << childExe << "\" "
        << "\"" << file.string() << "\" "
        << y_start << " "
        << region_height << " "
        << T << " "
        << X << " "
        << "\"" << shmName << "\"";
    std::string commandLine = oss.str();
    if (!fs::exists(childExe)) {
        std::cerr << ">>> ERROR: file not found on disk! <<<\n";
        return result;
    }
    // Copy into a mutable buffer (CreateProcessA may modify it)
    std::vector<char> cmdLineBuf(commandLine.begin(), commandLine.end());
    cmdLineBuf.push_back('\0');

    // Prepare ANSI startup info
    STARTUPINFOA siStartInfoA;
    ZeroMemory(&siStartInfoA, sizeof(siStartInfoA));
    siStartInfoA.cb = sizeof(siStartInfoA);

    const int    maxAttempts = 3;
    int          attempt = 0;
    DWORD        exitCode = 1;

    while (attempt < maxAttempts && exitCode != 0) {
        PROCESS_INFORMATION piProcInfo;
        ZeroMemory(&piProcInfo, sizeof(piProcInfo));

        BOOL bSuccess = CreateProcessA(
            /*lpApplicationName*/   NULL,
            /*lpCommandLine*/       cmdLineBuf.data(),
            /*lpProcessAttributes*/ NULL,
            /*lpThreadAttributes*/  NULL,
            /*bInheritHandles*/     TRUE,
            /*dwCreationFlags*/     0,
            /*lpEnvironment*/       NULL,
            /*lpCurrentDirectory*/  NULL,
            /*lpStartupInfo*/       &siStartInfoA,
            /*lpProcessInformation*/&piProcInfo
        );
        if (!bSuccess) {
            std::cerr << "CreateProcessA failed (" << GetLastError() << ")\n";
            CloseHandle(hMapFile);
            return result;
        }

        WaitForSingleObject(piProcInfo.hProcess, INFINITE);
        if (!GetExitCodeProcess(piProcInfo.hProcess, &exitCode)) {
            std::cerr << "Failed to get child exit code ("
                << GetLastError() << ")\n";
            exitCode = 1;
        }

        CloseHandle(piProcInfo.hProcess);
        CloseHandle(piProcInfo.hThread);

        if (exitCode != 0) {
            std::cerr << "Child returned " << exitCode
                << ", retrying (" << (attempt + 1)
                << "/" << maxAttempts << ")...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        ++attempt;
    }

    if (exitCode != 0) {
        std::cerr << "Child failed after "
            << maxAttempts << " attempts.\n";
        CloseHandle(hMapFile);
        return result;
    }

    // Map and copy data out
    LPVOID pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, totalBytes);
    if (!pBuf) {
        std::cerr << "Could not map view of file ("
            << GetLastError() << ")\n";
        CloseHandle(hMapFile);
        return result;
    }

    result.resize(totalElements);
    memcpy(result.data(), pBuf, totalBytes);

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return result;
}

std::vector<std::vector<double>> BasisManager::get_fk_region(int y_start, int y_end) {
    std::vector<std::vector<double>> fk;
    std::vector<fs::path> files;

    // Collect all .nc files with an underscore index
    for (auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto p = entry.path();
        if (p.extension() == ".nc" &&
            p.filename().string().find('_') != std::string::npos) {
            files.push_back(p);
        }
    }

    // Sort by the numeric suffix before ".nc"
    auto extractIndex = [](const fs::path& p) -> int {
        std::regex re("_(\\d+)\\.nc");
        std::smatch m;
        std::string name = p.filename().string();
        if (std::regex_search(name, m, re))
            return std::stoi(m[1].str());
        return std::numeric_limits<int>::max();
        };
    std::sort(files.begin(), files.end(),
        [&](auto& a, auto& b) { return extractIndex(a) < extractIndex(b); });

    // Launch up to 8 async readers, throttling to 6 active
    std::vector<std::future<std::vector<double>>> futures;
    for (auto& f : files) {
        // throttle
        while (futures.size() >= 8) {
            for (auto it = futures.begin(); it != futures.end(); ) {
                if (it->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    auto data = it->get();
                    if (!data.empty()) fk.push_back(std::move(data));
                    it = futures.erase(it);
                }
                else {
                    ++it;
                }
            }
            if (futures.size() >= 6)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        futures.emplace_back(
            std::async(std::launch::async,
                [f, y_start, y_end]() {
                    return read_nc_file(f.string(), y_start, y_end);
                })
        );
    }

    // gather remaining
    for (auto& fut : futures) {
        auto data = fut.get();
        if (!data.empty()) fk.push_back(std::move(data));
    }

    return fk;
}

std::vector<double> WaveManager::load_mariogramm_by_region(int y_start, int y_end) {
    return read_nc_file(nc_file, y_start, y_end);
}

// nc_reader_child.cpp
#define NOMINMAX 
#include <netcdf.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <windows.h>
#include <algorithm>
#include <stdexcept>
#include <string>

// ��������� ��� �������� ������������ ���������� �� NetCDF-�����
struct NcDimensions {
    size_t T;
    size_t Y;
    size_t X;
};

// ������� ������ ������������ ���������� "height" �� NetCDF-�����
NcDimensions read_nc_dimensions(int ncid, int varid) {
    NcDimensions dims;
    int dimids[3];
    int retval = nc_inq_vardimid(ncid, varid, dimids);
    if (retval != NC_NOERR) {
        throw std::runtime_error("������ ��������� ��������������� ���������: " + std::string(nc_strerror(retval)));
    }
    retval = nc_inq_dimlen(ncid, dimids[0], &dims.T);
    if (retval != NC_NOERR) {
        throw std::runtime_error("������ ������ ������� ��������� T: " + std::string(nc_strerror(retval)));
    }
    retval = nc_inq_dimlen(ncid, dimids[1], &dims.Y);
    if (retval != NC_NOERR) {
        throw std::runtime_error("������ ������ ������� ��������� Y: " + std::string(nc_strerror(retval)));
    }
    retval = nc_inq_dimlen(ncid, dimids[2], &dims.X);
    if (retval != NC_NOERR) {
        throw std::runtime_error("������ ������ ������� ��������� X: " + std::string(nc_strerror(retval)));
    }
    return dims;
}

int main(int argc, char* argv[]) {
    // ����� �������������: ��������� <filename> <y_start> <region_height> [<shmName>]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <filename> <y_start> <region_height> [<shmName>]" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int y_start = std::atoi(argv[2]);
    int region_height_arg = std::atoi(argv[3]);

    bool useSharedMemory = (argc >= 5);
    std::string shmName;
    if (useSharedMemory) {
        shmName = argv[4];
    }

    int ncid;
    int retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "������ �������� ����� " << filename << ": " << nc_strerror(retval) << std::endl;
        return 1;
    }

    int varid;
    retval = nc_inq_varid(ncid, "height", &varid);
    if (retval != NC_NOERR) {
        std::cerr << "���������� 'height' �� ������� � ����� " << filename << std::endl;
        nc_close(ncid);
        return 1;
    }

    // �������� ����������� �� �����
    NcDimensions dims;
    try {
        dims = read_nc_dimensions(ncid, varid);
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        nc_close(ncid);
        return 1;
    }

    // ������������� region_height: �� ��������� ���������� ���������� ����� � �����
    int region_height = std::min(region_height_arg, static_cast<int>(dims.Y) - y_start);
    size_t T = dims.T;
    size_t X = dims.X;
    size_t totalElements = T * region_height * X;

    // ������ ����� ������ ��� ��������� ����� � stdout
    HANDLE hMapFile = NULL;
    LPVOID pBuf = NULL;
    if (useSharedMemory) {
        // ��������� ���������� ��� ��� ����� ������ (����� �������� ������� ��������)
        DWORD pid = GetCurrentProcessId();
        DWORD tick = GetTickCount();
        shmName += "_" + std::to_string(pid) + "_" + std::to_string(tick);
        std::wstring wshmName(
            shmName.begin(),
            shmName.end()
        );
        hMapFile = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            static_cast<DWORD>(totalElements * sizeof(double)),
            wshmName.c_str()       // wide‐char pointer now
        );
        if (hMapFile == NULL) {
            std::cerr << "�� ������� ������� ������ ����������� ����� (" << GetLastError() << ")\n";
            nc_close(ncid);
            return 1;
        }
        pBuf = MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, totalElements * sizeof(double));
        if (pBuf == NULL) {
            std::cerr << "�� ������� ���������� ���� (" << GetLastError() << ")\n";
            CloseHandle(hMapFile);
            nc_close(ncid);
            return 1;
        }
    }

    // ��������� ������ �� NetCDF � �����
    size_t start[3] = { 0, static_cast<size_t>(y_start), 0 };
    size_t count[3] = { T, static_cast<size_t>(region_height), X };
    std::vector<double> buffer(totalElements);
    retval = nc_get_vara_double(ncid, varid, start, count, buffer.data());
    if (retval != NC_NOERR) {
        std::cerr << "������ ������ ������: " << nc_strerror(retval) << std::endl;
        nc_close(ncid);
        if (useSharedMemory) {
            UnmapViewOfFile(pBuf);
            CloseHandle(hMapFile);
        }
        return 1;
    }
    nc_close(ncid);

    // ������������������ ������:
    // �������� ������� (�� �����): [T][region_height][X]
    // �������� �������: [region_height][X][T]
    if (useSharedMemory) {
        double* dest = reinterpret_cast<double*>(pBuf);
        for (size_t t = 0; t < T; t++) {
            for (int y = 0; y < region_height; y++) {
                for (size_t x = 0; x < X; x++) {
                    size_t src_idx = t * (region_height * X) + y * X + x;
                    size_t dst_idx = y * (X * T) + x * T + t;
                    dest[dst_idx] = buffer[src_idx];
                }
            }
        }
        UnmapViewOfFile(pBuf);
        CloseHandle(hMapFile);
    }
    else {
        // ������: ������������������ � �������������� ����� � ����� � stdout
        std::vector<double> reordered(totalElements);
        for (size_t t = 0; t < T; t++) {
            for (int y = 0; y < region_height; y++) {
                for (size_t x = 0; x < X; x++) {
                    size_t src_idx = t * (region_height * X) + y * X + x;
                    size_t dst_idx = y * (X * T) + x * T + t;
                    reordered[dst_idx] = buffer[src_idx];
                }
            }
        }
        std::cout.write(reinterpret_cast<const char*>(reordered.data()), totalElements * sizeof(double));
        std::cout.flush();
    }

    return 0;
}

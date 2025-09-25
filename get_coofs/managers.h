#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <cstddef>
#include <vector>
#include "stable_data_structs.h"
#include <optional>

struct PointSpan {
    int y = 0;
    int x_start = 0;
    std::size_t length = 0;
    std::size_t offset = 0;
};

class WaveManager {
public:
    explicit WaveManager(const std::string& nc_file_);
    ~WaveManager();
    WaveManager(const WaveManager&) = delete;
    WaveManager& operator=(const WaveManager&) = delete;
    WaveManager(WaveManager&&) noexcept;
    WaveManager& operator=(WaveManager&&) noexcept;

    bool valid() const noexcept;
    const std::string& path() const noexcept { return nc_file; }
    bool get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const;

    std::vector<std::vector<double>> load_point_spans(
        const std::vector<PointSpan>& spans,
        int T,
        std::size_t total_points) const;

private:
    int ncid = -1;
    int varid = -1;
    std::size_t dims[3] = { 0, 0, 0 };
    std::string nc_file;
};

class BasisManager {
public:
    explicit BasisManager(const std::string& folder_);
    ~BasisManager();
    BasisManager(const BasisManager&) = delete;
    BasisManager& operator=(const BasisManager&) = delete;
    BasisManager(BasisManager&&) noexcept;
    BasisManager& operator=(BasisManager&&) noexcept;

    bool valid() const noexcept;
    std::size_t basis_count() const;
    bool get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const;
    const std::string& directory() const noexcept { return folder; }

    std::vector<std::vector<std::vector<double>>> load_point_spans(
        const std::vector<PointSpan>& spans,
        int T,
        std::size_t total_points) const;

private:
    struct BasisEntry {
        std::string path;
        int ncid = -1;
        int varid = -1;
        std::size_t dims[3] = { 0, 0, 0 };
    };

    std::vector<BasisEntry> bases;
    std::string folder;
};

#endif // MANAGERS_H

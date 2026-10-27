#include "common/TempDir.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "io/CheckpointIO.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

using namespace ac;

class CheckpointIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("ac_checkpoint_test");
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

    std::filesystem::path test_dir_;
};

TEST_F(CheckpointIOTest, WriteAndRead) {
    Grid grid(Dim3{8, 8, 8}, Spacing{0.5, 0.5, 0.5});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    // Fill with known data
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z) {
                phi(x, y, z) = std::sin(x * 0.1) * std::cos(y * 0.2) * std::sin(z * 0.3);
                u(x, y, z) = x + y * 0.1 + z * 0.01;
            }

    auto path = test_dir_ / "test_checkpoint.acbin";
    CheckpointIO::write(path, 42, 1.234, 0.005, grid, phi, u);

    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_TRUE(CheckpointIO::is_valid_checkpoint(path));

    auto data = CheckpointIO::read(path);
    EXPECT_EQ(data.step, 42);
    EXPECT_DOUBLE_EQ(data.time, 1.234);
    EXPECT_DOUBLE_EQ(data.dt, 0.005);
    EXPECT_EQ(data.grid.Nx(), 8);
    EXPECT_EQ(data.grid.Ny(), 8);
    EXPECT_EQ(data.grid.Nz(), 8);
    EXPECT_DOUBLE_EQ(data.grid.dx(), 0.5);

    // Verify field data integrity
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z) {
                EXPECT_DOUBLE_EQ(data.phi(x, y, z), phi(x, y, z));
                EXPECT_DOUBLE_EQ(data.u(x, y, z), u(x, y, z));
            }
}

TEST_F(CheckpointIOTest, InvalidFile) {
    auto path = test_dir_ / "not_a_checkpoint.bin";
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << "garbage data";
    }
    EXPECT_FALSE(CheckpointIO::is_valid_checkpoint(path));
    EXPECT_THROW(CheckpointIO::read(path), std::runtime_error);
}

TEST_F(CheckpointIOTest, MissingFile) {
    EXPECT_FALSE(CheckpointIO::is_valid_checkpoint("/nonexistent/file.acbin"));
}

TEST_F(CheckpointIOTest, LargerGrid) {
    Grid grid(Dim3{16, 16, 16}, Spacing{0.3, 0.3, 0.3});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    phi.fill(1.0);
    u.fill(-0.5);

    auto path = test_dir_ / "large.acbin";
    CheckpointIO::write(path, 100, 5.0, 0.01, grid, phi, u);

    auto data = CheckpointIO::read(path);
    EXPECT_EQ(data.step, 100);
    EXPECT_EQ(data.phi.size(), 16u * 16 * 16);

    for (std::size_t i = 0; i < data.phi.size(); ++i) {
        EXPECT_DOUBLE_EQ(data.phi.data()[i], 1.0);
        EXPECT_DOUBLE_EQ(data.u.data()[i], -0.5);
    }
}

// CRC-32/IEEE check value from the catalogue of parametrised CRC algorithms.
TEST_F(CheckpointIOTest, Crc32MatchesStandardCheckValue) {
    const char msg[] = "123456789";
    EXPECT_EQ(CheckpointIO::compute_crc32(msg, 9), 0xCBF43926u);
    EXPECT_EQ(CheckpointIO::compute_crc32(msg, 0), 0u);
}

// The checkpoint hashes phi then u in place; that must equal one pass over phi||u.
TEST_F(CheckpointIOTest, Crc32ChainingEqualsSinglePass) {
    std::vector<unsigned char> buf(1000);
    for (std::size_t i = 0; i < buf.size(); ++i)
        buf[i] = static_cast<unsigned char>((i * 131u + 7u) & 0xFFu);
    const uint32_t one_pass = CheckpointIO::compute_crc32(buf.data(), buf.size());
    for (std::size_t split : {std::size_t{0}, std::size_t{1}, std::size_t{499}, buf.size()}) {
        const uint32_t head = CheckpointIO::compute_crc32(buf.data(), split);
        const uint32_t chained =
            CheckpointIO::compute_crc32(buf.data() + split, buf.size() - split, head);
        EXPECT_EQ(chained, one_pass) << "split=" << split;
    }
}

// The stored checksum must cover exactly the field bytes that follow the header.
TEST_F(CheckpointIOTest, StoredCrcCoversFieldPayload) {
    Grid grid(Dim3{6, 5, 4}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    for (std::size_t i = 0; i < phi.size(); ++i) {
        phi.data()[i] = std::sin(0.01 * static_cast<double>(i));
        u.data()[i] = -0.8 + 1e-3 * static_cast<double>(i);
    }
    auto path = test_dir_ / "crc_payload.acbin";
    CheckpointIO::write(path, 7, 0.5, 0.01, grid, phi, u);

    std::ifstream ifs(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ASSERT_EQ(bytes.size(), sizeof(CheckpointIO::Header) + 2 * phi.size() * sizeof(Real));

    CheckpointIO::Header hdr{};
    std::memcpy(&hdr, bytes.data(), sizeof(hdr));
    const uint32_t payload_crc = CheckpointIO::compute_crc32(
        bytes.data() + sizeof(CheckpointIO::Header), bytes.size() - sizeof(CheckpointIO::Header));
    EXPECT_NE(hdr.data_crc32, 0u);
    EXPECT_EQ(hdr.data_crc32, payload_crc);
}

// A single flipped bit in the field payload must be rejected on restore.
TEST_F(CheckpointIOTest, CorruptedPayloadIsRejected) {
    Grid grid(Dim3{8, 8, 8}, Spacing{0.5, 0.5, 0.5});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    phi.fill(0.25);
    u.fill(-0.75);
    auto path = test_dir_ / "corrupt.acbin";
    CheckpointIO::write(path, 3, 0.1, 0.01, grid, phi, u);
    ASSERT_NO_THROW(CheckpointIO::read(path));

    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        const auto offset = static_cast<std::streamoff>(sizeof(CheckpointIO::Header) + 100);
        f.seekg(offset);
        char byte = 0;
        f.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x01);
        f.seekp(offset);
        f.write(&byte, 1);
    }
    EXPECT_THROW(CheckpointIO::read(path), std::runtime_error);
}

namespace {

/// read() must throw a runtime_error whose message contains `needle`.
void expect_read_rejects(const std::filesystem::path& path, const std::string& needle) {
    try {
        (void)CheckpointIO::read(path);
        ADD_FAILURE() << "accepted " << path;
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

} // namespace

// A checkpoint is exactly the header and the two fields of its grid: a file
// one byte longer or shorter is rejected (the payload CRC alone cannot catch
// trailing bytes; the resume helper rejects them too).
TEST_F(CheckpointIOTest, FileSizeMustMatchTheHeaderGrid) {
    Grid grid(Dim3{8, 8, 8}, Spacing{0.5, 0.5, 0.5});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(0.25);
    u.fill(-0.75);
    const auto longer = test_dir_ / "longer.acbin", shorter = test_dir_ / "shorter.acbin";
    CheckpointIO::write(longer, 3, 0.1, 0.01, grid, phi, u);
    CheckpointIO::write(shorter, 3, 0.1, 0.01, grid, phi, u);
    const auto size = std::filesystem::file_size(longer);
    ASSERT_EQ(size, sizeof(CheckpointIO::Header) + 2 * 512 * sizeof(double));
    std::ofstream(longer, std::ios::binary | std::ios::app).put('\0');
    std::filesystem::resize_file(shorter, size - 1);
    expect_read_rejects(longer, "does not match its 8x8x8 grid");
    expect_read_rejects(shorter, "does not match its 8x8x8 grid");
}

// The size check replaces a cap of 100000 cells per axis: a long thin grid is
// a valid checkpoint, and a header whose grid overflows 64 bits is rejected
// by its size before anything is allocated.
TEST_F(CheckpointIOTest, LongAxisIsReadAndAnOverflowingHeaderIsRejected) {
    Grid grid(Dim3{100001, 3, 3}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(0.5);
    u.fill(-0.25);
    const auto path = test_dir_ / "long_axis.acbin";
    CheckpointIO::write(path, 7, 0.7, 0.1, grid, phi, u);
    const auto data = CheckpointIO::read(path);
    EXPECT_EQ(data.grid.Nx(), 100001);
    EXPECT_EQ(data.phi.data()[data.phi.size() - 1], 0.5);

    CheckpointIO::Header hdr{};
    hdr.Nx = hdr.Ny = hdr.Nz = 1 << 30;
    hdr.dx = hdr.dy = hdr.dz = 0.4;
    hdr.num_fields = 2;
    const auto huge = test_dir_ / "huge.acbin";
    std::ofstream(huge, std::ios::binary)
        .write(reinterpret_cast<const char*>(&hdr), sizeof(hdr))
        .write("payload", 7);
    expect_read_rejects(huge, "overflows");
}

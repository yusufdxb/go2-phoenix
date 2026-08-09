// Copyright 2026 Yusuf Guenena. MIT License.
//
// Unitree LowCmd CRC.
//
// R19: this is NOT a standard CRC32. Polynomial 0x04C11DB7, init 0xFFFFFFFF,
// NO input reflection, NO output reflection, NO final XOR, and the inner loop
// XORs the polynomial on each set data bit. It is not zlib's CRC32 and it is
// not MPEG-2's either. Substituting a table-driven standard implementation
// produces a checksum the firmware rejects, so the loop is transcribed
// literally from motor_crc.py:101-117 rather than "modernised".
//
// R18: the CRC covers (sizeof(LowCmd) >> 2) - 1 little-endian uint32 words,
// i.e. every byte except the trailing crc field itself.
#ifndef PHOENIX_CORE__MOTOR_CRC_HPP_
#define PHOENIX_CORE__MOTOR_CRC_HPP_

#include <cstddef>
#include <cstdint>

namespace phoenix_core
{

// Byte size of the Unitree LowCmd struct. Asserted by the Python against its
// ctypes mirror; any change here is a wire-format change.
constexpr std::size_t kLowCmdSize = 812;

// CRC over a stream of uint32 words.
std::uint32_t crc32_core(const std::uint32_t * words, std::size_t count) noexcept;

// CRC over a LowCmd byte buffer, excluding the trailing 4-byte crc field.
// `len` must be kLowCmdSize.
std::uint32_t compute_lowcmd_crc(const std::uint8_t * bytes, std::size_t len) noexcept;

}  // namespace phoenix_core

#endif  // PHOENIX_CORE__MOTOR_CRC_HPP_

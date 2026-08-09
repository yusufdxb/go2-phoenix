// Copyright 2026 Yusuf Guenena. MIT License.
// See motor_crc.hpp. Literal port of motor_crc.py:101-127.
#include "phoenix_core/motor_crc.hpp"

#include <cstring>

namespace phoenix_core
{

std::uint32_t crc32_core(const std::uint32_t * words, std::size_t count) noexcept
{
  std::uint32_t crc = 0xFFFFFFFFu;
  constexpr std::uint32_t poly = 0x04C11DB7u;

  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t data = words[i];
    for (std::uint32_t xbit = 1u << 31; xbit != 0u; xbit >>= 1) {
      // Note the order: the shift/xor on the CRC happens FIRST, and the data
      // bit then conditionally xors the polynomial into the result. That
      // second xor is what makes this non-standard, and reordering the two
      // produces a plausible-looking checksum the firmware rejects.
      if (crc & 0x80000000u) {
        crc = (crc << 1) ^ poly;
      } else {
        crc = crc << 1;
      }
      if (data & xbit) {
        crc ^= poly;
      }
    }
  }
  return crc;
}

std::uint32_t compute_lowcmd_crc(const std::uint8_t * bytes, std::size_t len) noexcept
{
  if (bytes == nullptr || len != kLowCmdSize) {
    return 0u;
  }
  // (sizeof >> 2) - 1 words: everything except the trailing crc field.
  const std::size_t count = (len >> 2) - 1u;
  std::uint32_t crc = 0xFFFFFFFFu;
  constexpr std::uint32_t poly = 0x04C11DB7u;

  for (std::size_t i = 0; i < count; ++i) {
    // Little-endian load, done bytewise so the result does not depend on the
    // host's endianness or on alignment.
    std::uint32_t data = static_cast<std::uint32_t>(bytes[i * 4]) |
      (static_cast<std::uint32_t>(bytes[i * 4 + 1]) << 8) |
      (static_cast<std::uint32_t>(bytes[i * 4 + 2]) << 16) |
      (static_cast<std::uint32_t>(bytes[i * 4 + 3]) << 24);
    for (std::uint32_t xbit = 1u << 31; xbit != 0u; xbit >>= 1) {
      if (crc & 0x80000000u) {
        crc = (crc << 1) ^ poly;
      } else {
        crc = crc << 1;
      }
      if (data & xbit) {
        crc ^= poly;
      }
    }
  }
  return crc;
}

}  // namespace phoenix_core

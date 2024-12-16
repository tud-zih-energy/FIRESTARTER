/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include "firestarter/FunctionSelection.hpp"

#include <gtest/gtest.h>

namespace {

class FakeCpuFeatures : public firestarter::CpuFeatures {
  [[nodiscard]] auto hasAll(const CpuFeatures& /*Features*/) const -> bool override { return true; };
};

class FakeCpuModel : public firestarter::CpuModel {
  [[nodiscard]] auto operator<(const CpuModel& /*Other*/) const -> bool override { return true; }

  [[nodiscard]] auto operator==(const CpuModel& /*Other*/) const -> bool override { return true; }
};

template <bool PayloadIsAvailable> class FakePayload : public firestarter::payload::Payload {
public:
  FakePayload()
      : firestarter::payload::Payload(
            /*Name=*/"AvailablePayload", /*RegisterSize=*/0, /*RegisterCount=*/0) {}

  void init(double* /*MemoryAddr*/, uint64_t /*BufferSize*/) const override {}

  void lowLoadFunction(volatile firestarter::LoadThreadWorkType& /*LoadVar*/,
                       std::chrono::microseconds /*Period*/) const override {}

  [[nodiscard]] auto isAvailable(const firestarter::CpuFeatures& /*Features*/) const -> bool override {
    return PayloadIsAvailable;
  }

  [[nodiscard]] auto compilePayload(const firestarter::payload::PayloadSettings& /*Settings*/,
                                    bool
                                    /*DumpRegisters*/,
                                    bool /*ErrorDetection*/, bool /*PrintAssembler*/) const
      -> firestarter::payload::CompiledPayload::UniquePtr override {
    return {nullptr, nullptr};
  }

  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string> override { return {}; }
};

constexpr const unsigned ThreadCount1 = 1;
constexpr const unsigned ThreadCount2 = 2;
constexpr const unsigned InvalidThreadCount = 3;

template <bool PayloadIsAvailable, bool IsDefault = true>
class FakePlatfromConfig : public firestarter::platform::PlatformConfig {
private:
  unsigned Id;

public:
  explicit FakePlatfromConfig(unsigned Id)
      : firestarter::platform::PlatformConfig(
            /*Name=*/"PlatfromConfigTest",
            /*Settings=*/
            firestarter::payload::PayloadSettings(
                /*Threads=*/{ThreadCount1, ThreadCount2}, /*DataCacheBufferSize=*/{}, /*RamBufferSize=*/0,
                /*Lines=*/0,
                /*Groups=*/firestarter::InstructionGroups(firestarter::InstructionGroups::InternalType())),
            /*Payload=*/std::make_shared<FakePayload<PayloadIsAvailable>>())
      , Id(Id) {}

  [[nodiscard]] auto getId() const -> auto{ return Id; }

private:
  [[nodiscard]] auto isDefault(const firestarter::CpuModel& /*Model*/,
                               const firestarter::CpuFeatures& /*Features*/) const -> bool override {
    return IsDefault;
  }

  [[nodiscard]] auto clone() const -> std::unique_ptr<PlatformConfig> override { return nullptr; }

  [[nodiscard]] auto cloneConcreate(std::optional<unsigned> InstructionCacheSize, unsigned ThreadsPerCore) const
      -> std::unique_ptr<PlatformConfig> override {
    auto Config = std::make_unique<FakePlatfromConfig<PayloadIsAvailable, IsDefault>>(Id);
    Config->settings().concretize(InstructionCacheSize, ThreadsPerCore);
    return Config;
  }
};

constexpr const unsigned PayloadId1 = 4;
constexpr const unsigned PayloadId2 = 5;

constexpr const std::optional<unsigned> InstructionCacheSize = 6;

template <bool WithPlatformConfigs, bool WithFallbackPlatformConfigs, bool FirstPayloadIsAvailable,
          bool SecondPayloadIsAvailable, bool FirstPlatformIsDefault = true, bool SecondPlatformIsDefault = true>
class FunctionSelectionTest : public firestarter::FunctionSelection {
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> PlatformConfigs;
  std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>> FallbackPlatformConfigs;

public:
  FunctionSelectionTest() {
    if constexpr (WithPlatformConfigs) {
      PlatformConfigs = {
          std::make_shared<FakePlatfromConfig<FirstPayloadIsAvailable, FirstPlatformIsDefault>>(PayloadId1),
          std::make_shared<FakePlatfromConfig<SecondPayloadIsAvailable, SecondPlatformIsDefault>>(PayloadId2)};
    }
    if constexpr (WithFallbackPlatformConfigs) {
      FallbackPlatformConfigs = {
          std::make_shared<FakePlatfromConfig<FirstPayloadIsAvailable, FirstPlatformIsDefault>>(PayloadId1),
          std::make_shared<FakePlatfromConfig<SecondPayloadIsAvailable, SecondPlatformIsDefault>>(PayloadId2)};
    }
  }

  [[nodiscard]] auto selectAvailableFunction(unsigned FunctionId, std::optional<unsigned> InstructionCacheSize,
                                             bool AllowUnavailablePayload) const
      -> std::unique_ptr<firestarter::platform::PlatformConfig> {
    return firestarter::FunctionSelection::selectAvailableFunction(FunctionId, FakeCpuFeatures(), InstructionCacheSize,
                                                                   AllowUnavailablePayload);
  }

  [[nodiscard]] auto selectDefaultOrFallbackFunction(std::optional<unsigned> InstructionCacheSize,
                                                     unsigned NumThreadsPerCore) const
      -> std::unique_ptr<firestarter::platform::PlatformConfig> {
    return firestarter::FunctionSelection::selectDefaultOrFallbackFunction(
        FakeCpuModel(), FakeCpuFeatures(), "VendorString", "ModelString", InstructionCacheSize, NumThreadsPerCore);
  }

  [[nodiscard]] auto platformConfigs() const
      -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return PlatformConfigs;
  }

  [[nodiscard]] auto fallbackPlatformConfigs() const
      -> const std::vector<std::shared_ptr<firestarter::platform::PlatformConfig>>& override {
    return FallbackPlatformConfigs;
  }
};

template <bool FirstPayloadIsAvailable, bool SecondPayloadIsAvailable, bool FirstPlatformIsDefault = true,
          bool SecondPlatformIsDefault = true>
using FunctionSelectionWithPlatformConfigsTest =
    FunctionSelectionTest</*WithPlatformConfigs*/ true, /*WithFallbackPlatformConfigs*/ false, FirstPayloadIsAvailable,
                          SecondPayloadIsAvailable, FirstPlatformIsDefault, SecondPlatformIsDefault>;

template <bool FirstPayloadIsAvailable, bool SecondPayloadIsAvailable, bool FirstPlatformIsDefault = true,
          bool SecondPlatformIsDefault = true>
using FunctionSelectionWithFallbackPlatformConfigsTest =
    FunctionSelectionTest</*WithPlatformConfigs*/ false, /*WithFallbackPlatformConfigs*/ true, FirstPayloadIsAvailable,
                          SecondPayloadIsAvailable, FirstPlatformIsDefault, SecondPlatformIsDefault>;

template <bool FirstPayloadIsAvailable, bool SecondPayloadIsAvailable>
void checkSelectAvailableFunctionThrowWithPlatformConfigs(const unsigned FunctionId,
                                                          const std::optional<unsigned> InstructionCacheSize,
                                                          const bool AllowUnavailablePayload) {
  EXPECT_ANY_THROW((void)(FunctionSelectionWithPlatformConfigsTest<FirstPayloadIsAvailable, SecondPayloadIsAvailable>()
                              .selectAvailableFunction(
                                  /*FunctionId=*/FunctionId, InstructionCacheSize, AllowUnavailablePayload)));
}

/// Check that the function with the given id selects the correct available payload, thread count and instruction cache
/// size.
template <bool FirstPayloadIsAvailable, bool SecondPayloadIsAvailable, bool SelectedPayloadAvailable>
void checkSelectAvailableFunctionNoThrowWithPlatformConfigs(const unsigned FunctionId, const unsigned PayloadId,
                                                            const unsigned ThreadCount,
                                                            const std::optional<unsigned> InstructionCacheSize,
                                                            const bool AllowUnavailablePayload) {
  std::unique_ptr<firestarter::platform::PlatformConfig> Config;

  EXPECT_NO_THROW(Config =
                      (FunctionSelectionWithPlatformConfigsTest<FirstPayloadIsAvailable, SecondPayloadIsAvailable>()
                           .selectAvailableFunction(FunctionId, InstructionCacheSize, AllowUnavailablePayload)));

  const auto* FakeConfg = dynamic_cast<FakePlatfromConfig<SelectedPayloadAvailable>*>(Config.get());
  // Id describes the nth element in the list of platform configs.
  EXPECT_EQ(FakeConfg->getId(), PayloadId);
  EXPECT_EQ(FakeConfg->settings().thread(), ThreadCount);
  EXPECT_EQ(FakeConfg->settings().instructionCacheSize(), InstructionCacheSize);
}

/// Check that the correct fallback function is selected
template <bool FirstPayloadIsAvailable, bool SecondPayloadIsAvailable>
void checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs(
    const unsigned PayloadId, const unsigned InputThreadCount, const unsigned OutputThreadCount,
    const std::optional<unsigned> InstructionCacheSize) {
  std::unique_ptr<firestarter::platform::PlatformConfig> Config;
  EXPECT_NO_THROW(
      Config = (FunctionSelectionWithFallbackPlatformConfigsTest<FirstPayloadIsAvailable, SecondPayloadIsAvailable>()
                    .selectDefaultOrFallbackFunction(InstructionCacheSize,
                                                     /*NumThreadsPerCore=*/InputThreadCount)));
  {
    const auto* FakeConfg = dynamic_cast<FakePlatfromConfig<true>*>(Config.get());
    // Id describes the nth element in the list of platform configs.
    EXPECT_EQ(FakeConfg->getId(), PayloadId);
    EXPECT_EQ(FakeConfg->settings().thread(), OutputThreadCount);
    EXPECT_EQ(FakeConfg->settings().instructionCacheSize(), InstructionCacheSize);
  }
}

/// Check that the default fallback function is selected
template <bool FirstPlatformIsDefault, bool SecondPlatformIsDefault>
void checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs(
    const unsigned PayloadId, const unsigned ThreadCount, const std::optional<unsigned> InstructionCacheSize) {
  std::unique_ptr<firestarter::platform::PlatformConfig> Config;
  EXPECT_NO_THROW(
      Config =
          (FunctionSelectionWithPlatformConfigsTest</*FirstPayloadIsAvailable=*/true, /*SecondPayloadIsAvailable=*/true,
                                                    FirstPlatformIsDefault, SecondPlatformIsDefault>()
               .selectDefaultOrFallbackFunction(InstructionCacheSize,
                                                /*NumThreadsPerCore=*/ThreadCount)));
  {
    const auto* FakeConfg = dynamic_cast<FakePlatfromConfig<true>*>(Config.get());
    // Id describes the nth element in the list of platform configs.
    EXPECT_EQ(FakeConfg->getId(), PayloadId);
    EXPECT_EQ(FakeConfg->settings().thread(), ThreadCount);
    EXPECT_EQ(FakeConfg->settings().instructionCacheSize(), InstructionCacheSize);
  }
}

template <bool FirstPlatformIsDefault, bool SecondPlatformIsDefault>
void checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs(const unsigned ThreadCount,
                                                                  const std::optional<unsigned> InstructionCacheSize) {
  EXPECT_ANY_THROW((void)(FunctionSelectionWithPlatformConfigsTest</*FirstPayloadIsAvailable=*/true,
                                                                   /*SecondPayloadIsAvailable=*/true,
                                                                   FirstPlatformIsDefault, SecondPlatformIsDefault>()
                              .selectDefaultOrFallbackFunction(InstructionCacheSize,
                                                               /*NumThreadsPerCore=*/ThreadCount)));
}

} // namespace

TEST(FunctionSelectionTest, CheckInvalidFunctionIds) {
  // Id 0 is not valid.
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                       /*SecondPayloadIsAvailable=*/true>(
      /*FunctionId=*/0, InstructionCacheSize,
      /*AllowUnavailablePayload=*/true);
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                       /*SecondPayloadIsAvailable=*/true>(
      /*FunctionId=*/0, InstructionCacheSize,
      /*AllowUnavailablePayload=*/false);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      /*FunctionId=*/0, InstructionCacheSize,
      /*AllowUnavailablePayload=*/true);
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      /*FunctionId=*/0, InstructionCacheSize,
      /*AllowUnavailablePayload=*/false);

  // Id is too big.
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                       /*SecondPayloadIsAvailable=*/true>(
      /*FunctionId=*/5, InstructionCacheSize,
      /*AllowUnavailablePayload=*/true);
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                       /*SecondPayloadIsAvailable=*/true>(
      /*FunctionId=*/5, InstructionCacheSize,
      /*AllowUnavailablePayload=*/false);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      /*FunctionId=*/5, InstructionCacheSize,
      /*AllowUnavailablePayload=*/true);
  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      /*FunctionId=*/5, InstructionCacheSize,
      /*AllowUnavailablePayload=*/false);
}

TEST(FunctionSelectionTest, CheckFunctionIds) {
  // Ids starting from 1 are valid
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      1, PayloadId1, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      1, PayloadId1, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      1, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                         /*SecondPayloadIsAvailable=*/false,
                                                         /*SelectedPayloadAvailable=*/false>(
      1, PayloadId1, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      2, PayloadId1, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      2, PayloadId1, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      2, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                         /*SecondPayloadIsAvailable=*/false,
                                                         /*SelectedPayloadAvailable=*/false>(
      2, PayloadId1, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      3, PayloadId2, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      3, PayloadId2, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      3, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                         /*SecondPayloadIsAvailable=*/false,
                                                         /*SelectedPayloadAvailable=*/false>(
      3, PayloadId2, ThreadCount1, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      4, PayloadId2, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                         /*SecondPayloadIsAvailable=*/true,
                                                         /*SelectedPayloadAvailable=*/true>(
      4, PayloadId2, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/true);

  checkSelectAvailableFunctionThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                       /*SecondPayloadIsAvailable=*/false>(
      4, InstructionCacheSize, /*AllowUnavailablePayload=*/false);
  checkSelectAvailableFunctionNoThrowWithPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                         /*SecondPayloadIsAvailable=*/false,
                                                         /*SelectedPayloadAvailable=*/false>(
      4, PayloadId2, ThreadCount2, InstructionCacheSize, /*AllowUnavailablePayload=*/true);
}

TEST(FunctionSelectionTest, NoFallbackFound) {
  EXPECT_ANY_THROW((void)(FunctionSelectionWithFallbackPlatformConfigsTest</*FirstPayloadIsAvailable=*/false,
                                                                           /*SecondPayloadIsAvailable=*/false>()
                              .selectDefaultOrFallbackFunction(
                                  /*InstructionCacheSize=*/{},
                                  /*NumThreadsPerCore=*/1)));
}

TEST(FunctionSelectionTest, CheckFallbackThreadCount) {
  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/false>(
      PayloadId1, /*InputThreadCount=*/ThreadCount1, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId1, /*InputThreadCount=*/ThreadCount1, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/false>(
      PayloadId1, /*InputThreadCount=*/ThreadCount2, /*OutputThreadCount=*/ThreadCount2, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId1, /*InputThreadCount=*/ThreadCount2, /*OutputThreadCount=*/ThreadCount2, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/false>(
      PayloadId1, /*InputThreadCount=*/InvalidThreadCount, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/true,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId1, /*InputThreadCount=*/InvalidThreadCount, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId2, /*InputThreadCount=*/ThreadCount1, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId2, /*InputThreadCount=*/ThreadCount2, /*OutputThreadCount=*/ThreadCount2, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithFallbackPlatformConfigs</*FirstPayloadIsAvailable=*/false,
                                                                         /*SecondPayloadIsAvailable=*/true>(
      PayloadId2, /*InputThreadCount=*/InvalidThreadCount, /*OutputThreadCount=*/ThreadCount1, InstructionCacheSize);
}

TEST(FunctionSelectionTest, NoDefaultFound) {
  // non of the configs are the default, no fallbacks available -> throw
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                               /*SecondPlatformIsDefault=*/false>(ThreadCount1,
                                                                                                  InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                               /*SecondPlatformIsDefault=*/false>(ThreadCount2,
                                                                                                  InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                               /*SecondPlatformIsDefault=*/false>(InvalidThreadCount,
                                                                                                  InstructionCacheSize);
}

TEST(FunctionSelectionTest, DefaultThreadCountFound) {
  // default config with correct thread count found -> no throw
  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                                 /*SecondPlatformIsDefault=*/true>(
      PayloadId1, ThreadCount1, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                                 /*SecondPlatformIsDefault=*/true>(
      PayloadId1, ThreadCount2, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                                 /*SecondPlatformIsDefault=*/false>(
      PayloadId1, ThreadCount1, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                                 /*SecondPlatformIsDefault=*/false>(
      PayloadId1, ThreadCount2, InstructionCacheSize);

  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                                 /*SecondPlatformIsDefault=*/true>(
      PayloadId2, ThreadCount1, InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionNoThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                                 /*SecondPlatformIsDefault=*/true>(
      PayloadId2, ThreadCount2, InstructionCacheSize);
}

TEST(FunctionSelectionTest, DefaultThreadCountNotFound) {
  // default config found but not the correct number of thread, no fallbacks available -> throw
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                               /*SecondPlatformIsDefault=*/true>(InvalidThreadCount,
                                                                                                 InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/false,
                                                               /*SecondPlatformIsDefault=*/true>(InvalidThreadCount,
                                                                                                 InstructionCacheSize);
  checkSelectDefaultOrFallbackFunctionThrowWithPlatformConfigs</*FirstPlatformIsDefault=*/true,
                                                               /*SecondPlatformIsDefault=*/false>(InvalidThreadCount,
                                                                                                  InstructionCacheSize);
}
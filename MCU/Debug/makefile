################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include user/subdir.mk
-include tflite_micro/tensorflow/lite/micro/testing/subdir.mk
-include tflite_micro/tensorflow/lite/micro/memory_planner/subdir.mk
-include tflite_micro/tensorflow/lite/micro/kernels/portable_optimized/subdir.mk
-include tflite_micro/tensorflow/lite/micro/kernels/subdir.mk
-include tflite_micro/tensorflow/lite/micro/benchmarks/subdir.mk
-include tflite_micro/tensorflow/lite/micro/subdir.mk
-include tflite_micro/tensorflow/lite/kernels/internal/subdir.mk
-include tflite_micro/tensorflow/lite/kernels/subdir.mk
-include tflite_micro/tensorflow/lite/core/api/subdir.mk
-include tflite_micro/tensorflow/lite/c/subdir.mk
-include firmware/subdir.mk
-include cmsis/subdir.mk
-include bsp/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(ASM_DEPS)),)
-include $(ASM_DEPS)
endif
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := MCU
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
SECONDARY_FLASH += \
MCU.hex \

SECONDARY_SIZE += \
MCU.siz \


# All Target
all: main-build

# Main-build Target
main-build: MCU.elf secondary-outputs

# Tool invocations
MCU.elf: $(OBJS) $(USER_OBJS) makefile objects.mk $(OPTIONAL_TOOL_DEPS)
	@echo 'Building target: $@'
	@echo 'Invoking: GNU Arm Cross C++ Linker'
	arm-none-eabi-g++ -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O0 -ffunction-sections  -g -T "C:\Users\mabor\Downloads\water-level-identification\MCU/ldscripts/AT32F403AxG_FLASH.ld" -Xlinker --gc-sections -L"C:\Users\mabor\Downloads\water-level-identification\MCU\cmsis_dsp" -Wl,-Map,"MCU.map" --specs=nano.specs --specs=nosys.specs -o "MCU.elf" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

MCU.hex: MCU.elf makefile objects.mk $(OPTIONAL_TOOL_DEPS)
	@echo 'Invoking: GNU Arm Cross Create Flash Image'
	arm-none-eabi-objcopy -O ihex "MCU.elf"  "MCU.hex"
	@echo 'Finished building: $@'
	@echo ' '

MCU.siz: MCU.elf makefile objects.mk $(OPTIONAL_TOOL_DEPS)
	@echo 'Invoking: GNU Arm Cross Print Size'
	arm-none-eabi-size --format=berkeley "MCU.elf"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(SECONDARY_FLASH)$(SECONDARY_SIZE)$(ASM_DEPS)$(S_DEPS)$(S_UPPER_DEPS)$(C_DEPS)$(CPP_DEPS) MCU.elf
	-@echo ' '

secondary-outputs: $(SECONDARY_FLASH) $(SECONDARY_SIZE)

.PHONY: all clean dependents main-build

-include ../makefile.targets

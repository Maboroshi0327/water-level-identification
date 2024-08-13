/**
  **************************************************************************
  * @file     main.c
  * @version  v2.1.2
  * @date     2022-08-16
  * @brief    main program
  **************************************************************************
  *                       Copyright notice & Disclaimer
  *
  * The software Board Support Package (BSP) that is made available to
  * download from Artery official website is the copyrighted work of Artery.
  * Artery authorizes customers to use, copy, and distribute the BSP
  * software and its related documentation for the purpose of design and
  * development in conjunction with Artery microcontrollers. Use of the
  * software is governed by this copyright notice and the following disclaimer.
  *
  * THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS WITHOUT WARRANTIES,
  * GUARANTEES OR REPRESENTATIONS OF ANY KIND. ARTERY EXPRESSLY DISCLAIMS,
  * TO THE FULLEST EXTENT PERMITTED BY LAW, ALL EXPRESS, IMPLIED OR
  * STATUTORY OR OTHER WARRANTIES, GUARANTEES OR REPRESENTATIONS,
  * INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
  * FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.
  *
  **************************************************************************
  */

#include "at32f403a_407_clock.h"
#include "at32f403a_407_board.h"
#include "arm_math.h"
#include "ai.h"

#define FFTLEN 2048

int i, mode = 0, adc_cnt = 0;
uint32_t testtt1, testtt2;

extern arm_rfft_fast_instance_f32 arm_rfft_fast_sR_f32_len2048;
float buffer_real_in[FFTLEN];
float buffer_cmplx_out[FFTLEN];
float buffer_real_out[FFTLEN/2];
float max_value;
uint32_t max_index;

float result[5];

void gpio_config_button(void);
void gpio_config_mag(void);
void tmr1_config(void);
void adc_config(void);

int main(void)
{
  system_clock_config();
  delay_init();
  gpio_config_button();
  gpio_config_mag();
  tmr1_config();
  adc_config();
  ai_setup();

  while (1)
  {
    if (gpio_input_data_bit_read(GPIOA, GPIO_PINS_0) == SET && mode == 0)
    {
      delay_ms(1000);
      while (gpio_input_data_bit_read(GPIOA, GPIO_PINS_0) == SET);
      mode = 1;
      delay_ms(20);
    }
    if (mode == 1)
    {
      gpio_bits_write(GPIOE, GPIO_PINS_9, TRUE);
      delay_ms(100);
      gpio_bits_write(GPIOE, GPIO_PINS_9, FALSE);
      adc_cnt = 0;
      mode = 2;
    }
    if (mode == 3)
    {
      mode = 0;

      arm_rfft_fast_f32(&arm_rfft_fast_sR_f32_len2048, buffer_real_in, buffer_cmplx_out, 0);
      arm_cmplx_mag_f32(buffer_cmplx_out, buffer_real_out, FFTLEN);
      arm_max_f32(buffer_real_out + 18, 283, &max_value, &max_index);
      for (i = 18; i < 301; i++)
        buffer_real_out[i] = buffer_real_out[i] / max_value;

      SysTick->LOAD = (uint32_t) (2400000);
      SysTick->VAL = 0x00;
      SysTick->CTRL |= SysTick_CTRL_ENABLE_Msk;
      testtt1 = SysTick->VAL;

      ai_predict(buffer_real_out + 18, 283, result);

      testtt2 = testtt1 - SysTick->VAL;
      SysTick->CTRL &= ~SysTick_CTRL_ENABLE_Msk;
      SysTick->VAL = 0x00;
    }
  }
}

void gpio_config_button(void)
{
  gpio_init_type gpio_init_struct;

  /* enable the gpioa clock */
  crm_periph_clock_enable(CRM_GPIOA_PERIPH_CLOCK, TRUE);

  /* set default parameter */
  gpio_default_para_init(&gpio_init_struct);

  /* configure the gpio */
  gpio_init_struct.gpio_drive_strength = GPIO_DRIVE_STRENGTH_STRONGER;
  gpio_init_struct.gpio_out_type = GPIO_OUTPUT_PUSH_PULL;
  gpio_init_struct.gpio_mode = GPIO_MODE_INPUT;
  gpio_init_struct.gpio_pins = GPIO_PINS_0;
  gpio_init_struct.gpio_pull = GPIO_PULL_NONE;
  gpio_init(GPIOA, &gpio_init_struct);
}
void gpio_config_mag(void)
{
  gpio_init_type gpio_init_struct;

  /* enable the gpioa clock */
  crm_periph_clock_enable(CRM_GPIOE_PERIPH_CLOCK, TRUE);

  /* set default parameter */
  gpio_default_para_init(&gpio_init_struct);

  /* configure the gpio */
  gpio_init_struct.gpio_drive_strength = GPIO_DRIVE_STRENGTH_STRONGER;
  gpio_init_struct.gpio_out_type = GPIO_OUTPUT_PUSH_PULL;
  gpio_init_struct.gpio_mode = GPIO_MODE_OUTPUT;
  gpio_init_struct.gpio_pins = GPIO_PINS_9;
  gpio_init_struct.gpio_pull = GPIO_PULL_DOWN;
  gpio_init(GPIOE, &gpio_init_struct);
}
void tmr1_config(void)
{
  tmr_output_config_type tmr_oc_init_structure;
  crm_periph_clock_enable(CRM_TMR1_PERIPH_CLOCK, TRUE);
  tmr_base_init(TMR1, 99, 240);
  tmr_cnt_dir_set(TMR1, TMR_COUNT_UP);
  tmr_clock_source_div_set(TMR1, TMR_CLOCK_DIV1);

  tmr_output_default_para_init(&tmr_oc_init_structure);
  tmr_oc_init_structure.oc_mode = TMR_OUTPUT_CONTROL_PWM_MODE_A;
  tmr_oc_init_structure.oc_output_state = TRUE;
  tmr_output_channel_config(TMR1, TMR_SELECT_CHANNEL_1, &tmr_oc_init_structure);

  tmr_channel_value_set(TMR1, TMR_SELECT_CHANNEL_1, 9);
  tmr_output_channel_buffer_enable(TMR1, TMR_SELECT_CHANNEL_1, TRUE);

  tmr_counter_enable(TMR1, TRUE);
  tmr_channel_enable(TMR1, TMR_SELECT_CHANNEL_1, TRUE);
  tmr_output_enable(TMR1, TRUE);
}
void adc_config(void)
{
  gpio_init_type gpio_initstructure;
  adc_base_config_type adc_base_struct;

  /* enable the adc1 and gpio clock */
  crm_periph_clock_enable(CRM_ADC1_PERIPH_CLOCK, TRUE);
  crm_periph_clock_enable(CRM_GPIOA_PERIPH_CLOCK, TRUE);

  /* configure the adc_ch1 pin */
  gpio_default_para_init(&gpio_initstructure);
  gpio_initstructure.gpio_mode = GPIO_MODE_ANALOG;
  gpio_initstructure.gpio_pins = GPIO_PINS_1;
  gpio_init(GPIOA, &gpio_initstructure);

  /* configure the adc1 */
  crm_adc_clock_div_set(CRM_ADC_DIV_8);
  nvic_priority_group_config(NVIC_PRIORITY_GROUP_4);
  nvic_irq_enable(ADC1_2_IRQn, 1, 0);

  /* select ordinary mode */
  adc_combine_mode_select(ADC_INDEPENDENT_MODE);
  adc_base_default_para_init(&adc_base_struct);
  adc_base_struct.sequence_mode = FALSE;
  adc_base_struct.repeat_mode = FALSE;
  adc_base_struct.data_align = ADC_RIGHT_ALIGNMENT;
  adc_base_config(ADC1, &adc_base_struct);
  adc_ordinary_channel_set(ADC1, ADC_CHANNEL_1, 1, ADC_SAMPLETIME_239_5);
  adc_ordinary_conversion_trigger_set(ADC1, ADC12_ORDINARY_TRIG_TMR1CH1, TRUE);
  adc_interrupt_enable(ADC1, ADC_CCE_INT, TRUE);
  adc_enable(ADC1, TRUE);
}

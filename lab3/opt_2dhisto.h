#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* d_input, uint8_t* d_bins);

/* Include below the function headers of any other functions that you implement */
void opt_2dhisto_setup(uint32_t*& d_input, uint32_t** input, uint8_t*& d_bins, uint8_t* kernel_bins);
void opt_2dhisto_teardown(uint32_t*& d_input, uint8_t*& d_bins, uint8_t* kernel_bins);
uint32_t* AllocateDeviceInputArray(uint32_t **input);
uint8_t* AllocateDeviceBinsArray(uint8_t *kernel_bins);
void CopyInputToDevice(uint32_t*& d_input, uint32_t** input);
void CopyBinsToDevice(uint8_t *d_bins, uint8_t *kernel_bins);
void CopyBinsFromDevice(uint8_t *kernel_bins, uint8_t *d_bins);

#endif

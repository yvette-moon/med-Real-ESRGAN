# med-Real-ESRGAN

这一版能跑dicom，改成dicom的逻辑是

物理值映射 (HU Conversion)：利用 RescaleSlope 和 RescaleIntercept 将原始像素值线性转换为 CT 的 Hounsfield Unit (HU值)。

灰度空间处理 (Photometric Interpretation)：针对 MONOCHROME1 和 MONOCHROME2 的差异进行了反转处理。

窗宽窗位映射 (Windowing)：这是最核心的视觉映射步骤。

多通道扩展：将单通道灰度图复制为 3 通道 (RGB)，以适配 Real-ESRGAN 的网络结构

def build_psnr(ground_truth, estimate):
    # PSNR defination: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    import numpy
    import math

    if numpy.max(ground_truth) == 0:
        ground_truth = ground_truth * 255.0
    else:
        ground_truth = ground_truth / numpy.max(ground_truth) * 255.0

    if numpy.max(estimate) == 0:
        estimate = estimate * 255.0
    else:
        estimate = estimate / numpy.max(estimate) * 255.0

    mse = numpy.mean((ground_truth - estimate) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX ** 2 / mse)
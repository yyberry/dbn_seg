def dice_coeff(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) /
            (iflat.sum() + tflat.sum() + smooth))

def dice_loss(input, target):
    return 1 - dice_coeff(input, target)
from typing import DefaultDict
import numpy as np

from tqdm.auto import tqdm
from cv2        import resize
import cv2
from collections import defaultdict
import os

######################################################################
IMG_HW = 768
######################################################################

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(IMG_HW*IMG_HW, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(IMG_HW,IMG_HW).T


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_paired_data(df, dir_prefix, WIDTH, HEIGHT, augmentation=None):
    # load image
    img_id = df[0][0]
    image_path =os.path.join(dir_prefix, img_id)
    try:
        image = cv2.imread( image_path )
    except:
        image = np.zeros((IMG_HW, IMG_HW, 3), dtype=np.uint8)

    image = resize(image, (HEIGHT,WIDTH))
    #image = image.astype(np.float32)
    #image = image/cfg.NORM_FACTOR #- 1.0

    mask = np.zeros((IMG_HW, IMG_HW, 1))
    for local_df in df:
        mask_rle = local_df[1]
        if mask_rle is not np.nan:
            mask[:,:,0] += rle_decode(mask_rle)

    if augmentation:
        augmented = augmentation(image=image, mask=mask)
        image = augmented['image']
        mask  = augmented['mask']

    mask  = resize(mask.reshape(IMG_HW,IMG_HW), (HEIGHT,WIDTH)).reshape((HEIGHT,WIDTH,1))
    return image, mask

def batch_data_get(csv_reader, dir_prefix, batch_size, WIDTH, HEIGHT, augmentation=None):
    
    #name_idx_df = [dict(item)['ImageId'] for item in csv_reader]
    #name_unique_idx_df = []
    #for val in name_idx_df:
    #    if val in name_unique_idx_df: 
    #        continue 
    #    else:
    #        name_unique_idx_df.append(val)

    df ={}
    for item in csv_reader:
        if dict(item)['ImageId'] in df:
            df [dict(item)['ImageId']].append((dict(item)['ImageId'], dict(item)['EncodedPixels'], dict(item)['withShip'], dict(item)['npixel']))
        else:
            df [dict(item)['ImageId']] = []
            df [dict(item)['ImageId']].append((dict(item)['ImageId'], dict(item)['EncodedPixels'], dict(item)['withShip'], dict(item)['npixel']))
    #print (df)


    #snp.random.shuffle(img_ids)
    X =[]
    Y= []
    #for idx in tqdm( range(0, n_imgs, batch_size) ):
    #for idx in tqdm( range(0, 64, batch_size) ):
    #for idx in tqdm (range(0, 1)):
    for idx in  range(0, 1):
    
        #batch_x = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 3) )
        #batch_y = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 1) )
        end_idx = idx + batch_size
        #print (idx, end_idx)
        batch_img_ids = [v for v in list(df.keys())[idx:end_idx]]
        #print (batch_img_ids)
        
        for i,img_id in enumerate(batch_img_ids):
            img_df = df [img_id]
            x, y = load_paired_data(img_df, dir_prefix, WIDTH, HEIGHT, augmentation=augmentation)
            #batch_x[i] += x
            #batch_y[i] += y

            X.append (x)
            Y.append (y)

    return np.array (X), np.array(Y)

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    Yi[Yi>1] = 1
    
    TP = np.sum(Yi * y_predi)
    FP = np.sum((1-Yi) * y_predi)
    FN = np.sum(Yi * (1-y_predi))
    num = TP
    denum = float(TP + FP + FN)
    if (denum != 0.0):
        IoU = num/denum
    else:
        IoU = 0.0

    #print("TP: ", TP, "FP: ", FP, "FN: ", FN, "IoU", IoU)

    return IoU

def IoU_all(Yi_list,y_predi_list):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    for idx in range (Yi_list.shape[0]):
        Yi = Yi_list[idx]
        y_predi = y_predi_list [idx]

        Yi[Yi>1] = 1
        TP = np.sum(Yi * y_predi)
        FP = np.sum((1-Yi) * y_predi)
        FN = np.sum(Yi * (1-y_predi))

        num = TP
        denum = float(TP + FP + FN)
        if (denum != 0.0):
            IoU = num/denum
        else:
            IoU = 0.0

        IoUs.append(IoU)


    mIoU = np.mean(IoUs)

    return mIoU
import os, sys
import numpy  as np
from multiprocessing import Pool
from tqdm.auto import tqdm

from cv2        import resize
from skimage.io import imread     as skiImgRead
from segmentation_models.backbones import get_preprocessing

from config import fcn_config as cfg
######################################################################
IMG_HW    = 768
HEIGHT = cfg.HEIGHT
WIDTH  = cfg.WIDTH
BACKBONE  = 'resnet34'
######################################################################

preprocess_input = get_preprocessing(BACKBONE)

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

def count_pix_inpool(df_col):
    pool = Pool()
    res = pool.map( count_pix, df_col.items() )
    pool.close()
    pool.join()
    return res

def count_pix(row):
    v = row[1]
    if v is np.nan or type(v) != str: 
        return v
    else:
        return rle_decode(v).sum()

def batch_data_gen(csv_df, dir_prefix, batch_size, augmentation=None):
    name_idx_df = csv_df.set_index('ImageId')

#     img_ids = name_idx_df.index.unique().to_numpy()
    img_ids = np.array( name_idx_df.index.unique().tolist() )

    n_imgs  = img_ids.shape[0]
    
    while True:
        np.random.shuffle(img_ids)
        for idx in range(0, n_imgs, batch_size):
            batch_x = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 3) )
            batch_y = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 1) )

            end_idx = idx + batch_size
            batch_img_ids = img_ids[idx:end_idx]
            
            for i,img_id in enumerate(batch_img_ids):
                img_df = name_idx_df.loc[[img_id]]
                x, y = load_paired_data(img_df, dir_prefix, augmentation=augmentation)
                batch_x[i] += x
                batch_y[i] += y
            
            yield batch_x, batch_y



def load_paired_data(df, dir_prefix, augmentation=None):
    img_id = df.index.unique()[0]

    try:
        image = preprocess_input( skiImgRead( os.path.join(dir_prefix, img_id) ) )
    except:
        image = preprocess_input( np.zeros((IMG_HW, IMG_HW, 3), dtype=np.uint8) )

    mask = np.zeros((IMG_HW, IMG_HW, 1))
    for _,mask_rle in df['EncodedPixels'].iteritems():
        if mask_rle is np.nan:
            continue
        mask[:,:,0] += rle_decode(mask_rle)

    if augmentation:
        augmented = augmentation(image=image, mask=mask)
        image = augmented['image']
        mask  = augmented['mask']
    
    image = resize(image, (HEIGHT,WIDTH))
    mask  = resize(mask.reshape(IMG_HW,IMG_HW), (HEIGHT,WIDTH)).reshape((HEIGHT,WIDTH,1))
    return image, mask


def batch_data_get(csv_df, dir_prefix, batch_size, augmentation=None):
    name_idx_df = csv_df.set_index('ImageId')
    img_ids = np.array( name_idx_df.index.unique().tolist() )
    n_imgs  = img_ids.shape[0]

    print ("reading the test files")

    np.random.shuffle(img_ids)
    X =[]
    Y= []
    #for idx in tqdm( range(0, n_imgs, batch_size) ):
    for idx in tqdm( range(0, 64, batch_size) ):
    
        #batch_x = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 3) )
        #batch_y = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 1) )

        end_idx = idx + batch_size
        batch_img_ids = img_ids[idx:end_idx]
        
        for i,img_id in enumerate(batch_img_ids):
            img_df = name_idx_df.loc[[img_id]]
            x, y = load_paired_data(img_df, dir_prefix, augmentation=augmentation)
            #batch_x[i] += x
            #batch_y[i] += y


            X.append (x)
            Y.append (y)

    return np.array (X), np.array(Y)

def batch_data_get_all(csv_df, dir_prefix, batch_size, augmentation=None):
    name_idx_df = csv_df.set_index('ImageId')
    img_ids = np.array( name_idx_df.index.unique().tolist() )
    n_imgs  = img_ids.shape[0]

    print ("reading the test files")

    np.random.shuffle(img_ids)
    X =[]
    Y= []
    for idx in tqdm( range(0, n_imgs, batch_size) ):
    
        #batch_x = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 3) )
        #batch_y = np.zeros( (batch_size,) + (HEIGHT, WIDTH, 1) )

        end_idx = idx + batch_size
        batch_img_ids = img_ids[idx:end_idx]
        
        for i,img_id in enumerate(batch_img_ids):
            img_df = name_idx_df.loc[[img_id]]
            x, y = load_paired_data(img_df, dir_prefix, augmentation=augmentation)
            #batch_x[i] += x
            #batch_y[i] += y


            X.append (x)
            Y.append (y)

    return np.array (X), np.array(Y)
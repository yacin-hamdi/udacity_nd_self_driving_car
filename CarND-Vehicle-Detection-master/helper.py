import glob 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
import numpy as np


def convert_color(img, conv="RGB2YCrCb"):
    if conv == "RGB2YCrCb":
        print(conv)
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == "LUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == "HLS":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == "YUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    


def read_images(path):
    images_path = glob.glob(path)
    images = []
    for img_path in images_path:
        images.append((img_path.split('\\')[-1], mpimg.imread(img_path)))
    return images



def show_images(images, cmap=None):
    f, axes = plt.subplots(len(images)//2, 2, figsize=(12,12))
    indexes = range(len(images) * 2)
    for ax, index in zip(axes.flat, indexes):
        if index < len(images):
            path, image = images[index]
            ax.imshow(image, cmap=cmap)
            ax.set_title(path)
            ax.axis('off')
            
def bin_spatial(img, size=(32,32)):
    # rbin = cv2.resize(img[:,:,0], size).ravel()
    # gbin = cv2.resize(img[:,:,1], size).ravel()
    # bbin = cv2.resize(img[:,:,2], size).ravel()
    # features = np.hstack((rbin, gbin, bbin))
    # return features
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0,256)):
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    return hist_features


def get_hog_feature(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    
    if vis:
        feature, hog_image = hog(img, orientations=orient,
                                 pixels_per_cell=(pix_per_cell,pix_per_cell),
                                 cells_per_block=(cell_per_block, cell_per_block),
                                 block_norm='L2-Hys',
                                 transform_sqrt=True,
                                 visualize=vis, feature_vector=feature_vec)
        return feature, hog_image
    
    else:
        feature = hog(img, orientations=orient, 
                      pixels_per_cell=(pix_per_cell, pix_per_cell), 
                      cells_per_block=(cell_per_block, cell_per_block),
                      block_norm='L2-Hys',
                      transform_sqrt=True,
                      visualize=vis, feature_vector=feature_vec)
        return feature
    
    
def single_img_features(img, color_space="RGB", spatial_size=(32, 32), 
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, 
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    img_feature = []
    if color_space != "RGB":
        feature_image = convert_color(img, color_space)
    else:
        feature_image = np.copy(img)
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, spatial_size)
        img_feature.append(spatial_features)
    
    if hist_feat:
        color_features = color_hist(feature_image, nbins=hist_bins)
        img_feature.append(color_features)
        
    
    if hog_feat:
        if hog_channel == "ALL":
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_feature = get_hog_feature(feature_image[:,:,channel], orient=orient, 
                                              pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                             vis=False, feature_vec=False)
        else:
            hog_features = get_hog_feature(feature_image[:,:,hog_channel], orient=orient, 
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          vis=False, feature_vec=False)
        img_feature.append(hog_features)
        
    return np.concatenate(img_feature)





def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int32(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int32(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int32(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int32(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int32((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int32((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = extract_features(img, cspace=color_space, 
                                             size=spatial_size, hist_bins=hist_bins,
                                             orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block,
                                             hog_channel=hog_channel)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_feature(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_feature(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_feature(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img







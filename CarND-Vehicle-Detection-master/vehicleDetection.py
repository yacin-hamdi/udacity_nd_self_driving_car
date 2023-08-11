import cv2
from skimage.feature import hog
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


class VehicleDetection:
    def __init__(self, color_space="YCrCb", orient=8, pix_per_cell=8, 
                 cell_per_block=2, hog_channel="ALL", spatial_size=(16,16), 
                 hist_bins=32, hist_range=(0,256)):
        
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.hist_range = hist_range
        
        self.X_scaler = StandardScaler()
        self.clf = LinearSVC()
        
    def extract_features(self, img):


        # apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:,:,channel],  
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = self.get_hog_features(feature_image[:,:,hog_channel], 
                                            vis=False, feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        spatial_features = self.bin_spatial(feature_image)

        # Apply color_hist() 
        hist_features = self.color_hist(feature_image)

        return np.concatenate((spatial_features, hist_features, hog_features))
    
    
    
    def scale_features(self, cars_features, notcars_features):
        X = np.vstack((cars_features, notcars_features)).astype(np.float64)
        y = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))

        rand_state = np.random.randint(0,100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

        self.X_scaler = self.X_scaler.fit(X_train)
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
    
    def fit_model(self, X_train, y_train):
        time1 = time.time()
        self.clf.fit(X_train, y_train)
        time2 = time.time()
        return round(time2-time1, 2)
        
        
    def bin_spatial(self, img):
        # rbin = cv2.resize(img[:,:,0], self.spatial_size).ravel()
        # gbin = cv2.resize(img[:,:,1], self.spatial_size).ravel()
        # bbin = cv2.resize(img[:,:,2], self.spatial_size).ravel()
        # features = np.hstack((rbin, gbin, bbin))
        # return features
        bin_features = cv2.resize(img, self.spatial_size).ravel() 
        # Return the feature vector
        return bin_features
    
    
    def color_hist(self, img):
        rhist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.hist_range)
        ghist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.hist_range)
        bhist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.hist_range)

        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        return hist_features


    def get_hog_features(self, img, vis=False, feature_vec=True):

        if vis:
            feature, hog_image = hog(img, orientations=self.orient,
                                     pixels_per_cell=(self.pix_per_cell,self.pix_per_cell),
                                     cells_per_block=(self.cell_per_block, self.cell_per_block),
                                     transform_sqrt=True,
                                     visualize=vis, feature_vector=feature_vec)
            return feature, hog_image

        else:
            feature = hog(img, orientations=self.orient, 
                          pixels_per_cell=(self.pix_per_cell, self.pix_per_cell), 
                          cells_per_block=(self.cell_per_block, self.cell_per_block),
                          transform_sqrt=True,
                          visualize=vis, feature_vector=feature_vec)
            return feature
        
    def convert_color(self, img, conv="YCrCb"):
        if conv == "YCrCb":
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif conv == "HSV":
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif conv == "LUV":
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif conv == "HLS":
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif conv == "YUV":
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, window=64):

        draw_img = np.copy(img)

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step 
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step 

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, feature_vec=False)
        hog2 = self.get_hog_features(ch2, feature_vec=False)
        hog3 = self.get_hog_features(ch3, feature_vec=False)
        
        
        
        found_cars = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = self.bin_spatial(subimg)
                hist_features = self.color_hist(subimg)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))  
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    found_cars.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

        return found_cars
    
    def add_heat(self, heatmap, bbox_list):
        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heatmap
    
    def apply_threshold(self, heatmap, threshold=1):
        heatmap[heatmap <= threshold] = 0
        return heatmap
    
    def draw_labeled_box(self, img, labels):
        for car_number in range(1, labels[1]+1):
            nonzero = (car_number == labels[0]).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 6)
        return img
    
    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

import numpy as np 
import cv2
import matplotlib.pyplot as plt


def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def abs_sobel_thres(img, orient='x', kernel=3, threshold=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(hls, cv2.CV_64F, 1,0, ksize=kernel) 
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_s = np.zeros_like(scaled_sobel)
    binary_s[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    
    
    return binary_s
def mag_gradient(img, kernel=3, threshold=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(hls, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(hls, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_sobelxy = np.sqrt(np.power(sobelx, 2)+np.power(sobely, 2))
    scaled_gradient = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_gradient)
    binary_output[(scaled_gradient >= threshold[0])&(scaled_gradient <= threshold[1])] = 1
    return binary_output

def dir_gradient(img, kernel=3, threshold=(0.7, 1,3)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(hls, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(hls, cv2.CV_64F, 0, 1, ksize=kernel)
    dir_sobelxy = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(dir_sobelxy)
    binary_output[(dir_sobelxy >= threshold[0]) & (dir_sobelxy <= threshold[1])] = 1
    return binary_output

def color_threshold(img, threshold=(10, 200)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:,:,2]
    binary_output = np.zeros_like(L)
    binary_output[(L >=threshold[0]) & (L <=threshold[1])] = 1
    return binary_output

def gray_threshold(img, threshold=(208, 255)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(img)
    binary_output[(img >= threshold[0]) & (img <= threshold[1])] = 1
    return binary_output

def crop_image(img, offset):
    return img[(img.shape[0]//2) + offset:, :]


def warp_corners(c_img):
    offset = 100
    # c_img = crop_image(img, offset)
    img_size = (c_img.shape[1], c_img.shape[0])
    
    
    src = np.float32([[(img_size[0])//2-offset, (img_size[1])//2+offset], 
                      [(img_size[0])//2+offset, (img_size[1])//2+offset], 
                      [0, img_size[1]], 
                      [img_size[0], img_size[1]]])

    dst = np.float32([[0, 0], 
                    [img_size[0], 0],
                    [offset, img_size[1]],   
                    [img_size[0]-offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(c_img, M, img_size)
    return warped, M, Minv



def undistort_img(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs  = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    u_img = cv2.undistort(img, mtx, dist, None, mtx)
    return u_img, mtx, dist




def find_lanes(binary_image):
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/700
    
    histogram = np.sum(binary_image[binary_image.shape[0]//2:, :], axis=0)
    out_image = np.dstack((binary_image, binary_image, binary_image))
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # plt.plot(histogram)
    # plt.plot(midpoint, 0, '.')
    # plt.plot(leftx_base, 0, '.')
    # plt.plot(rightx_base, 0, '.')
    nwindows = 9
    margin = 100
    minpix = 50
    
    window_height = np.int32(binary_image.shape[0]//nwindows)
    nonzero = binary_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        win_y_low = binary_image.shape[0] - (window+1)*window_height
        win_y_high = binary_image.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # cv2.rectangle(out_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255, 0), 2)
        # cv2.rectangle(out_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
         ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
     # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    out_image[lefty, leftx] = [255, 0, 0]
    out_image[righty, rightx] = [0, 0, 255]
    
    # return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)

    return left_fit, right_fit, left_fit_m, right_fit_m, out_image





def draw_rectangle(u_img,warp_img, Minv, left_fit, right_fit):
    
    yMax = u_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (u_img.shape[1], u_img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(u_img, 1, newwarp, 0.3, 0)
    return result


def calculateCurvature(y_eval, left_fit_cr, right_fit_cr):
    
    ym_per_pix = 30/720 
    left_curvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = (1+ (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5/np.abs(2*right_fit_cr[0])
    return left_curvature, right_curvature


def drawCurvature(img, left_curv, right_curv):
    img_cp = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    cv2.putText(img_cp, 'Left curvature: {:.0f} m'.format(left_curv), (50, 50), font, 2, fontColor, 2)
    cv2.putText(img_cp, 'Right curvature: {:.0f} m'.format(right_curv), (50, 120), font, 2, fontColor, 2)
    return img_cp
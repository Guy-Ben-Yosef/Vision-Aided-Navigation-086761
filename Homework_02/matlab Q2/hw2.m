%% 1. a)

% K matrix.
K = [480 0   320;
     0   480 270;
     0   0   1];

% Rotation and translation matrix [R|t].
T_G_C = [0.5363 -0.8440 0 -451.2459;
         0.8440  0.5363 0  257.0322;
         0       0      1  400;];

P = K * T_G_C;

%% 1. b)

% 3D point in global frame.
X_G = [ 350;
       -250;
        -35;
         1;];

% Image coordinates in homogeneous form.
i_tilde = P * X_G;

i_normalized = i_tilde ./ i_tilde(3);

%% 1. c)

z = [241.5;
     169;
     1;];

err = norm(z - i_normalized);

fprintf('Error in pixels: %f\n', err);

%% 2. d)

% Extacting the K matrix, principal point and focal length.

cam_K = cameraParams.K
cam_princ_point = cameraParams.PrincipalPoint
cam_foc_len = cameraParams.FocalLength

%% 3

vl_setup

image1 = imread('img1.jpeg');
image2 = imread('img2.jpeg');

% Extract SIFT features
[features1, valid_points1] = extractSIFTFeatures(image1);
[features2, valid_points2] = extractSIFTFeatures(image2);
REM Runs one experiment for CSS 487 Program 4
REM parameters: ImageDirectory NumImages imagenames... NumDescriptors descriptornames... homographies...
REM The number of homographies should be one less than the number of images.
REM (Each homography is with respect to the first image.)

start /w ../Debug/Program4.exe ./images/bark/ 3 img1.ppm img2.ppm img3.ppm 2 SIFT SURF H1to2p.txt H1to3p.txt

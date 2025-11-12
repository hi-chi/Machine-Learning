
# --------- problem geometry
l1 = 0.07 # length from source to center of first magnet
l2 = 0.23 # length from source to center of second magnet
l3 = 0.15 # length from source to first screen
l4 = 0.40 # length from source to second screen
rmag1 = 0.06 # radius of magnet1
rmag2 = 0.06 # radius of magnet2

# screen geometry 1
x_min1=-0.01
x_max1=0.09
y_min1=-0.02
y_max1=0.02
dx1=6.5e-05
nx1=int((x_max1-x_min1)/dx1)
ny1=int((y_max1-y_min1)/dx1)

# screen geometry 2
x_min2=-0.0
x_max2=0.12
y_min2=-0.035
y_max2=0.035
dx2=8.1e-05
nx2=int((x_max2-x_min2)/dx2)
ny2=int((y_max2-y_min2)/dx2)

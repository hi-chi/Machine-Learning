
# --------- problem geometry
l1 = 0.275  # length from source to center of first magnet
l2 = 0.53  # length from source to center of second magnet
l3 = 0.32  # length from source to first screen
l4 = 0.675  # length from source to second screen
rmag = 0.03  # radius of magnet

# screen geometry 1
x_min1 = -0.005
x_max1 = 0.015
y_min1 = -0.003
y_max1 = 0.003
dx1 = 0.00002
nx1 = int((x_max1 - x_min1) / dx1)
ny1 = int((y_max1 - y_min1) / dx1)

# screen geometry 2
x_min2 = -0.005
x_max2 = 0.1
y_min2 = -0.02
y_max2 = 0.02
dx2 = 0.0001
nx2 = int((x_max2 - x_min2) / dx2)
ny2 = int((y_max2 - y_min2) / dx2)

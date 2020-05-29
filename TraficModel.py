#Initializing parameters and variables
total_num_drivers = 4000.0
alpha =.22
beta = 8.5
individual_lane_capacity = 2100.0
distance = 10.0
speed_slow_lane = 55.0
speed_medium_lane = 58.0
speed_fast_lane = 62.0

#######################################################################
#Solving for best response dynamics
######################################################################



import random
#First Lane (slow 55 mph)
def f1(x):
    return ((distance/speed_slow_lane)*(1+alpha*((x/individual_lane_capacity)**beta)))

#Second Lane (medium 58 mph)
def f2(x):
    return ((distance/speed_medium_lane)*(1+alpha*((x/individual_lane_capacity)**beta)))
#Third Lane (fast 62 mph)

def f3(x):
    return ((distance/speed_fast_lane)*(1+alpha*((x/individual_lane_capacity)**beta)))


def best_response_dynamics():
    #Number in lane one
    lane_one_traffic = 0
    #Number in lane two
    lane_two_traffic = 0
    #Number in lane three
    lane_three_traffic = 0


    #Randomly initializing the drivers to the lane
    for i in range(int(total_num_drivers)):
        lane = random.randint(1,3)
        if(lane==1):
            lane_one_traffic = lane_one_traffic + 1
        elif(lane==2):
            lane_two_traffic = lane_two_traffic + 1
        else:
            lane_three_traffic = lane_three_traffic + 1
    l_1_start = lane_one_traffic
    l_2_start = lane_two_traffic
    l_3_start = lane_three_traffic

    #Initial Cost
    lane_one_traffic_cost = f1(lane_one_traffic)
    lane_two_traffic_cost = f2(lane_two_traffic)
    lane_three_traffic_cost = f3(lane_three_traffic)


    def be_better(l1, l2, l3):
        #If a driver in lane one can switch to lane 2 or 3 and be better off, return true and lane option
        if((f1(l1)>f2(l2+1)) or f1(l1) > f3(l3 + 1)):
            if(l1>0):
                return (True, 1)
        #If a driver in lane two can switch to lane 1 or 3 and be better off, return true and lane option
        if ((f2(l2) > f1(l1 + 1)) or f2(l2) > f3(l3 + 1)):
            if(l2>0):
                return (True, 2)
        #If a driver in lane three can switch to lane 1 or 2 and be better off, return true and lane option
        if ((f3(l3) > f1(l1 + 1)) or f3(l3) > f2(l2 + 1)):
            if(l3>0):
                return (True, 3)
        return (False, 0)

    continue_switching = be_better(lane_one_traffic, lane_two_traffic, lane_three_traffic)[0]
    count = 1
    while(continue_switching):
        l = [1,2,3]
        current_lane = be_better(lane_one_traffic, lane_two_traffic, lane_three_traffic)[1]
        l.remove(current_lane)
        if(current_lane==1):
            lane_one_traffic = lane_one_traffic - 1
            #If moving to lane 2 is better, then move to lane 2
            if((f2(lane_two_traffic+1)) < (f3(lane_three_traffic+1))):
                lane_two_traffic = lane_two_traffic + 1
                lane_two_traffic_cost = f2(lane_two_traffic)
            #Else move to lane 3
            else:
                lane_three_traffic = lane_three_traffic + 1
                lane_two_traffic_cost = f3(lane_three_traffic)
        elif(current_lane==2):
            lane_two_traffic = lane_two_traffic - 1
            #If moving to lane 1 is better, then move to lane 1
            if((f1(lane_one_traffic + 1)) < (f3(lane_three_traffic + 1))):
                lane_one_traffic = lane_one_traffic + 1
                lane_one_traffic_cost = f1(lane_one_traffic)
            #Else move to lane 3
            else:
                lane_three_traffic = lane_three_traffic + 1
                lane_three_traffic_cost = f3(lane_three_traffic)
        else:
            lane_three_traffic = lane_three_traffic - 1
            #If moving to lane 1 is better, then move to lane 1
            if((f1(lane_one_traffic + 1)) < (f2(lane_two_traffic + 1))):
                lane_one_traffic = lane_one_traffic + 1
                lane_one_traffic_cost = f1(lane_one_traffic_cost)
            #Else move to lane 2
            else:
                lane_two_traffic = lane_two_traffic + 1
                lane_two_traffic_cost = f2(lane_two_traffic_cost)
        continue_switching = be_better(lane_one_traffic, lane_two_traffic, lane_three_traffic)[0]
        traffic = (lane_one_traffic, lane_two_traffic, lane_three_traffic)
        traffic_2 = (f1(lane_one_traffic), f2(lane_two_traffic), f3(lane_three_traffic))
        if(count%100==0):
            print(str(count) + ": "+ str(traffic))
        count = count + 1
    print(str(count) + ": "+ str(traffic))
    print(traffic_2)

    print('Following are results for SELFISH traffic placement: ')
    print('lane one traffic: ' + str(lane_one_traffic))
    print('lane two traffic: ' + str(lane_two_traffic))
    print('lane three traffic: ' + str(lane_three_traffic))
    best_response_traffic = [lane_one_traffic, lane_two_traffic, lane_three_traffic]
    return(best_response_traffic)



##########################################################################
#Solving for social optimal
#########################################################################

#Defining objective function
def objective(x):
    x1 = int(x[0])
    x2 = int(x[1])
    x3 = int(x[2])
    return (x1*((distance / speed_slow_lane)*(1 + alpha * ((x1 / individual_lane_capacity) ** beta)))+
            x2*((distance / speed_medium_lane)*(1 + alpha * ((x2 / individual_lane_capacity) ** beta))) +
            x3*((distance / speed_fast_lane) * (1 + alpha * ((x3 / individual_lane_capacity) ** beta))))

from gekko import GEKKO

def optimum_solution():
    m  = GEKKO()
    m.options.SOLVER = 1
    lane_1 = m.Var(integer = True, lb=0, ub = individual_lane_capacity)
    lane_2 = m.Var(integer = True, lb=0, ub = individual_lane_capacity)
    lane_3 = m.Var(integer = True, lb=0, ub = individual_lane_capacity)
    m.Minimize(lane_1*((distance / speed_slow_lane)*(1 + alpha * ((lane_1 / individual_lane_capacity) ** beta)))+
                lane_2*((distance / speed_medium_lane)*(1 + alpha * ((lane_2 / individual_lane_capacity) ** beta))) +
                lane_3*((distance / speed_fast_lane) * (1 + alpha * ((lane_3 / individual_lane_capacity) ** beta))))
    m.Equation(lane_1 + lane_2 + lane_3 == total_num_drivers)
    m.solve(disp = False)
    print('Following are results for OPTIMUM traffic placement: ')
    print('lane one traffic: '+ str(lane_1.value[0]))
    print('lane two traffic: '+ str(lane_2.value[0]))
    print('lane three traffic: '+ str(lane_3.value[0]))
    opt_traffic = [lane_1.value[0],lane_2.value[0], lane_3.value[0] ]
    print(f1(opt_traffic[0]), f2(opt_traffic[1]), f3(opt_traffic[2]))
    return(opt_traffic)



#######################################################################
#Price of Anarchy (PoA
######################################################################


best_response_traffic = best_response_dynamics()
opt_traffic = optimum_solution()

selfish_trafic_cost = objective(best_response_traffic)

opt_traffic_cost = objective(opt_traffic)

PoA=selfish_trafic_cost/opt_traffic_cost

print('Selfish traffic cost is: ' + str(selfish_trafic_cost))
print('Optimal traffic cost is: ' + str(opt_traffic_cost))


print('Price of anarchy is: ' + str(PoA))

#########################################################
#Sensitivity analysis
########################################################


####################################
#Varying alpha and Beta
###################################
#Creating the data to graph
import numpy as np
X_temp = np.linspace(0.05, 0.8, 20)
Y_temp = np.linspace(4, 12, 20)
X_temp, Y_temp = np.meshgrid(X_temp, Y_temp)
print(X_temp)
print(Y_temp)
Z_temp = []

for (x_i, y_i) in zip(X_temp, Y_temp):
    temp = []
    for (p,q) in zip(x_i,y_i):
        alpha = p
        beta = q
        print(alpha, beta)
        best_response_traffic = best_response_dynamics()
        opt_traffic = optimum_solution()

        selfish_trafic_cost = objective(best_response_traffic)

        opt_traffic_cost = objective(opt_traffic)

        PoA=selfish_trafic_cost/opt_traffic_cost
        temp.append(PoA)
    Z_temp.append(temp)

#Graphing
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
#%matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.close()
fig = plt.figure()
ax = fig.gca(projection='3d')
Z_temp = np.asarray(Z_temp)

# Plot the surface.
surf = ax.plot_surface(X_temp, Y_temp, Z_temp, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_title("PoA with Varying Beta and Alpha", fontdict=None)

# Customize the z axis.
ax.set_zlim(1.01, 1.07)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
####################################
#Varying total number of drivers
###################################

# Make data.
num_drivers= np.arange(1, 6300, 25)
poas = []
for i in num_drivers:
    total_num_drivers = i
    best_response_traffic = best_response_dynamics()
    opt_traffic = optimum_solution()

    selfish_trafic_cost = objective(best_response_traffic)

    opt_traffic_cost = objective(opt_traffic)
    PoA=selfish_trafic_cost/opt_traffic_cost
    print(PoA)
    poas.append(PoA)

#graphing

plt.close()
plt.plot(num_drivers, poas, color = 'r')

plt.xlabel('Total Number of Drivers')
plt.ylabel('PoA')
plt.title('Total Number Drivers vs PoA')
plt.show()


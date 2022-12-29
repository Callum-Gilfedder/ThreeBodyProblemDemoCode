#Three body problem
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#Initial Conditions

#Masses of bodies
massdifference = 0.001
m1 = 1.0 + massdifference
m2 = 1.0 + massdifference
m3 = 1.0 + massdifference
G = 1.0

x3dot = -0.93240737
y3dot = -0.86473146
posdifference = 0.00 #used to alter inital positions.
veldifference = 0.00#used to alter intiial velocities.

#Initial conditions of b1
r1x = 0.97000436 
r1y = -0.24308753  
v1x = x3dot/(-2.0) + veldifference
v1y = y3dot/(-2.0) 

#Initial conditions of b2 
r2x = -0.97000436 
r2y = 0.24308753  
v2x = x3dot/(-2.0) + veldifference
v2y = y3dot/(-2.0) 

#Initial conditions of b3
r3x = 0.0 
r3y = 0.0 
v3x = x3dot + veldifference
v3y = y3dot 

#So we have the bodies initial positions and velocities

t = 0
period = 6.32591398
number_of_timesteps = 400.0
h = period/number_of_timesteps
repetitions = 1.0
t_end = period*repetitions
k = 1.5


plt.figure(1)
plt.title("Experimenting with 3-body problem")
plt.grid(True)

#Body 1
r1 = np.array([r1x,r1y])
v1 = np.array([v1x,v1y])

#Body 2
r2 = np.array([r2x,r2y])
v2 = np.array([v2x,v2y])

#Body 3
r3 = np.array([r3x,r3y])
v3 = np.array([v3x,v3y])

def rxy3(r1,r2):
    rxy3 = (((r2[0] - r1[0])**2) + ((r2[1] - r1[1])**2))**1.5
    return rxy3

def F(r1):
    a1x = ((G*m2*(r2[0]-r1[0]))/(rxy3(r1,r2))) + ((G*m3*(r3[0]-r1[0]))/(rxy3(r1,r3)))
    a1y = ((G*m2*(r2[1]-r1[1]))/(rxy3(r1,r2))) + ((G*m3*(r3[1]-r1[1]))/(rxy3(r1,r3)))
    A = np.array([a1x,a1y])
    return A

def F2(r2):
    a2x = (-(G*m1*(r2[0]-r1[0]))/(rxy3(r1,r2))) + ((G*m3*(r3[0]-r2[0]))/(rxy3(r2,r3)))
    a2y = (-(G*m1*(r2[1]-r1[1]))/(rxy3(r1,r2))) + ((G*m3*(r3[1]-r2[1]))/(rxy3(r2,r3)))
    A = np.array([a2x,a2y])
    return A

def F3(r3):
    a3x = (-(G*m1*(r3[0]-r1[0]))/(rxy3(r1,r3))) + (-(G*m2*(r3[0]-r2[0]))/(rxy3(r2,r3)))
    a3y = (-(G*m1*(r3[1]-r1[1]))/(rxy3(r1,r3))) + (-(G*m2*(r3[1]-r2[1]))/(rxy3(r2,r3)))
    A = np.array([a3x,a3y])
    return A

#These forces below are edited versions of above for the RK4 algorithm that concatenates the r and v arrays
#into one larger array w. They should be calculating the exact same thing though.

w0 = np.array([0.97000436, -0.24308753, x3dot/(-2.0), y3dot/(-2.0), -0.97000436, 0.24308753, x3dot/(-2.0), y3dot/(-2.0), 0.0, 0.0, x3dot, y3dot])
w  = np.array([r1x, r1y, v1x, v1y, r2x, r2y, v2x, v2y, r3x, r3y, v3x, v3y])

def RF1(w1,w2,w3):
    a1x = ((G*m2*(w2[0]-w1[0]))/(rxy3(w1,w2))) + ((G*m3*(w3[0]-w1[0]))/(rxy3(w1,w3)))
    a1y = ((G*m2*(w2[1]-w1[1]))/(rxy3(w1,w2))) + ((G*m3*(w3[1]-w1[1]))/(rxy3(w1,w3)))
    j = np.array([w1[2],w1[3],a1x,a1y])
    return j

def RF2(w1,w2,w3):
    a2x = (-(G*m1*(w2[0]-w1[0]))/(rxy3(w1,w2))) + ((G*m3*(w3[0]-w2[0]))/(rxy3(w2,w3)))
    a2y = (-(G*m1*(w2[1]-w1[1]))/(rxy3(w1,w2))) + ((G*m3*(w3[1]-w2[1]))/(rxy3(w2,w3)))
    j = np.array([w2[2],w2[3],a2x,a2y])
    return j

def RF3(w1,w2,w3):
    a3x = (-(G*m1*(w3[0]-w1[0]))/(rxy3(w1,w3))) + (-(G*m2*(w3[0]-w2[0]))/(rxy3(w2,w3)))
    a3y = (-(G*m1*(w3[1]-w1[1]))/(rxy3(w1,w3))) + (-(G*m2*(w3[1]-w2[1]))/(rxy3(w2,w3)))
    j = np.array([w3[2],w3[3],a3x,a3y])
    return j

def RF(w):
    w1, w2, w3 = np.split(w,3)
    j1 = RF1(w1,w2,w3)
    j2 = RF2(w1,w2,w3)
    j3 = RF3(w1,w2,w3)
    return np.concatenate((j1,j2,j3))

def RK4(w):
    k1 = RF(w)
    k2 = RF(w + 0.5*k1*h)
    k3 = RF(w + 0.5*k2*h)
    k4 = RF(w + k3*h)
    RK4 = h*(1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return RK4

def mag(x,y):
    mag = (x**2 + y**2)**(0.5)
    return mag

def mag2(x1,y1,x2,y2):
    mag2 = (((x2 - x1)**2) + ((y2 - y1)**2))**(0.5)
    return mag2
    

def Hamiltonian(w):
    k1 = 0.5*m1*((mag(w[2],w[3]))**2)
    k2 = 0.5*m2*((mag(w[6],w[7]))**2)
    k3 = 0.5*m3*((mag(w[10],w[11]))**2)
    T = (k1 + k2 + k3)
    u1 = -G*(m2*m1*(1/mag2(w[0],w[1],w[4],w[5])) + m3*m1*(1/mag2(w[0],w[1],w[8],w[9])))
    u2 = -G*(m1*m2*(1/mag2(w[0],w[1],w[4],w[5])) + m3*m2*(1/mag2(w[4],w[5],w[8],w[9])))
    u3 = -G*(m1*m3*(1/mag2(w[0],w[1],w[8],w[9])) + m2*m3*(1/mag2(w[4],w[5],w[8],w[9])))
    U = u1 + u2 + u3
    H = T + U
    return H

def Angular(w):
    r1, v1 ,r2 ,v2 ,r3 ,v3 = np.split(w, 6) 
    a1 = np.cross(m1*r1, v1)
    a2 = np.cross(m2*r2, v2)
    a3 = np.cross(m3*r3, v3)
    a = a1 + a2 + a3
    A = np.linalg.norm(a)
    return A
    

n = 0
nsteps = np.round(t_end/h)

plt.figure(1)
plt.grid(True)
#Original orbit
r1x_list = []
r1y_list = []
v1x_list = []
v1y_list = []

r2x_list = []
r2y_list = []
v2x_list = []
v2y_list = []

r3x_list = []
r3y_list = []
v3x_list = []
v3y_list = []



energy_list = []
e_error_list = []

angular_list = []
angular_error_list = []

#adds initial conditions to lists

r1x_list.append(r1x)
r1y_list.append(r1y)
v1x_list.append(v1x)
v1y_list.append(v1y)  
r2x_list.append(r2x)
r2y_list.append(r2y)
v2x_list.append(v2x)
v2y_list.append(v2y)
r3x_list.append(r3x)
r3y_list.append(r3y)
v3x_list.append(v3x)
v3y_list.append(v3y)


t_list = []

def w_distance(r1x1, r1x2, r1y1, r1y2 ,v1x1, v1x2 , v1y1, v1y2, r2x1, r2x2 ,r2y1, r2y2, v2x1, v2x2, v2y1, v2y2, r3x1, r3x2, r3y1, r3y2, v3x1, v3x2, v3y1, v3y2):
    #Body 1
    a = (r1x2 - r1x1)**2
    b = (r1y2 - r1y1)**2
    c = (v1x2 - v1x1)**2
    d = (v1y2 - v1y1)**2
    #Body 2
    e = (r2x2 - r2x1)**2
    f = (r2y2 - r2y1)**2
    g = (v2x2 - v2x1)**2
    h = (v2y2 - v2y1)**2
    #Body 3
    i = (r3x2 - r3x1)**2
    j = (r3y2 - r3y1)**2
    k = (v3x2 - v3x1)**2
    l = (v3y2 - v3y1)**2
    #distance
    w_distance = (a+b+c+d+e+f+g+h+i+j+k+l)**(1/2)
    return w_distance


while n < nsteps:
    w = w + RK4(w)
    energy = Hamiltonian(w)
    energy_list.append(energy)
    e_error =  np.abs((Hamiltonian(w) - Hamiltonian(w0)))
    e_error_list.append(e_error)
    angular = Angular(w)
    angular_error = np.abs(Angular(w) - Angular(w0))
    angular_list.append(angular)
    angular_error_list.append(angular_error)
    r1x_list.append(w[0])
    r1y_list.append(w[1])
    v1x_list.append(w[2])
    v1y_list.append(w[3])
    r2x_list.append(w[4])
    r2y_list.append(w[5])
    v2x_list.append(w[6])
    v2y_list.append(w[7])
    r3x_list.append(w[8])
    r3y_list.append(w[9])
    v3x_list.append(w[10])
    v3y_list.append(w[11])
    t = t + h 
    t_list.append(t)
    n = n + 1

t = 0
c = 0.01
s = 0
period2 = 0
while t < nsteps:
    d =  mag2(r1x_list[t],r1y_list[t],r1x_list[t+1],r1y_list[t+1])
    u =  mag(v1x_list[t],v1y_list[t])
    period2 = period2 + d/u
    d2 = ((r1x_list[t+1] - r1x_list[0])**2 + (r1y_list[t+1] - r1y_list[0])**2)**(1/2)
    if t > 30: #Avoids returning values at start
        if d2 <= c:
            print ("Period = ", period2)
        else:
            print("Aperiodic =", period2)
    t = t + 1





t = 0
c = 0.01
s = 0
period2 = 0
distance = 0
while t < nsteps:
    d =  mag2(r1x_list[t],r1y_list[t],r1x_list[t+1],r1y_list[t+1])
    distance = distance + d
    u =  mag(v1x_list[t],v1y_list[t])
    period2 = period2 + d/u
    d2 = ((r1x_list[t+1] - r1x_list[0])**2 + (r1y_list[t+1] - r1y_list[0])**2)**(1/2)
    if t > 30: #Avoids returning values at start
        if d2 <= c:
            print ("Period = ", period2)
        else:
            print("Aperiodic =", period2)
    t = t + 1


#k = 4.0 moved this parameter to the top for easier adjustments.

plt.figure(1)
plt.title("Figure 8 Solution Perturbation")
plt.plot(r1x_list,r1y_list,color = "red",linewidth = 1)
plt.plot(r2x_list,r2y_list,color = "green", linewidth = 1)
plt.plot(r3x_list,r3y_list,color = "blue", linewidth = 1)
plt.axis("scaled")  
plt.xlim(-k,k)
plt.ylim(-k,k)


#plt.figure(4)
#plt.plot(t_list, e_error_list, color = "blue")
#plt.grid(True)

#plt.figure(5)
#plt.title("Angular Momentum")
#plt.plot(t_list, angular_list, color = "orange")
#plt.grid(True)

#plt.figure(6)
#plt.title("Energy")
#x = [t_list[0], t_list[-1]] 
#y = [Hamiltonian(w0), Hamiltonian(w0)]
#plt.grid(True)
#plt.plot(t_list, energy_list, color = "red")
#plt.plot(x, y, color = "black")
#plt.ylim(-20,20)

#plt.figure(7)
#plt.title("Error In The Energy Of The System")
#plt.plot(t_list, e_error_list, "red")
#plt.xlabel("Time")
#plt.ylabel("Error In Energy")
#plt.grid(True)

#plt.figure(8)
#plt.title("Error In The Angular Momentum Of The System")
#plt.plot(t_list, angular_error_list, "red")
#plt.xlabel("Time")
#plt.ylabel("Error In Angular Momentum")
#plt.grid(True)

#fig = plt.figure(2)
#ax = plt.subplot(1,1,1)
#plt.axis("scaled")
 
#def init_func():
#    ax.clear()
#    plt.xlabel("X")
#    plt.ylabel("Y")

#j = 1.5

#def update_plot(i):
#    ax.plot(r1x_list[0:i], r1y_list[0:i], color = "red", linewidth = 0.25)
#    ax.plot(r2x_list[0:i], r2y_list[0:i], color = "green", linewidth = 0.25)
#    ax.plot(r3x_list[0:i], r3y_list[0:i], color = "blue", linewidth = 0.25)
#    plt.xlim(-j,j)
#    plt.ylim(-j,j)
#    plt.grid(True)

m1e = ((r1x_list[-1] - r1x_list[0])**2 + (r1y_list[-1] - r1y_list[0])**2)**(1/2)
m2e = ((r2x_list[-1] - r2x_list[0])**2 + (r2y_list[-1] - r2y_list[0])**2)**(1/2)
m3e = ((r3x_list[-1] - r3x_list[0])**2 + (r3y_list[-1] - r3y_list[0])**2)**(1/2)
average_error = (m1e + m2e + m3e)/3
    
print("initial position error m1 =", m1e, "initial position error m2 =", m2e,"initial position error m3 =", m3e, "average error =", average_error) 

#plt.axis("scaled")
#anim = FuncAnimation(fig, update_plot, frames = np.arange(0,len(r1x_list),30), init_func = init_func, interval = 5)
#plt.show()

#anim.save(r"C:\Users\User\Desktop\drifting.gif", dpi = 150, fps = 30, writer="ffmpeg")

#Now I need a second perturbed orbit and I need to take the difference
#between the two orbits.

#PERTURBED SOLUTION BELOW
#PERTURBED SOLUTION BELOW
#PERTURBED SOLUTION BELOW
#PERTURBED SOLUTION BELOW
#PERTURBED SOLUTION BELOW


n2 = 0.0
#Masses of bodies
mp1 = 1.0
mp2 = 1.0
massdifferencep = 0.00
mp3 = 1.0 + massdifferencep
Gp = 1.0

xp3dot = -0.93240737
yp3dot = -0.86473146
posdifferencep = 0.00
veldifferencep = 0.00

#Initial conditions of b1
rp1x = 0.97000436 + posdifferencep
rp1y = -0.24308753
vp1x = xp3dot/(-2.0) + veldifferencep
vp1y = yp3dot/(-2.0)

#Initial conditions of b2 
rp2x = -0.97000436
rp2y = 0.24308753
vp2x = xp3dot/(-2.0)
vp2y = yp3dot/(-2.0)

#Initial conditions of b3
rp3x = 0.0
rp3y = 0.0
vp3x = xp3dot
vp3y = yp3dot

wp0 = np.array([-0.93240737, -0.86473146, 0.0, 0.0, -0.97000436, 0.24308753, xp3dot/(-2.0), yp3dot/(-2.0), 0.0, 0.0, xp3dot, yp3dot])
wp  = np.array([rp1x, rp1y, vp1x, vp1y, rp2x, rp2y, vp2x, vp2y, rp3x, rp3y, vp3x, vp3y])

#So we have the bodies initial positions and velocities

tp = 0.0
periodp = 6.32591398
#period = 6.32591398 (Better period I think)
number_of_timestepsp = number_of_timesteps
print(number_of_timestepsp)
print(number_of_timesteps)
hp = periodp/number_of_timestepsp
repetitionsp = repetitions
t_endp = periodp*repetitionsp
n2 = 0.0
nstepsp = np.round(t_endp/hp)

#Perturbed orbit
rp1x_list = []
rp1y_list = []
vp1x_list = []
vp1y_list = []

rp2x_list = []
rp2y_list = []
vp2x_list = []
vp2y_list = []

rp3x_list = []
rp3y_list = []
vp3x_list = []
vp3y_list = []

rp1x_list.append(rp1x)
rp1y_list.append(rp1y)
vp1x_list.append(vp1x)
vp1y_list.append(vp1y)  
rp2x_list.append(rp2x)
rp2y_list.append(rp2y)
vp2x_list.append(vp2x)
vp2y_list.append(vp2y)
rp3x_list.append(rp3x)
rp3y_list.append(rp3y)
vp3x_list.append(vp3x)
vp3y_list.append(vp3y)

tp_list = []

def rxy3p(rp1,rp2):
    rxy3p = (((rp2[0] - rp1[0])**2) + ((rp2[1] - rp1[1])**2))**1.5
    return rxy3p

def RFp1(wp1,wp2,wp3):
    ap1x = ((Gp*mp2*(wp2[0]-wp1[0]))/(rxy3p(wp1,wp2))) + ((Gp*mp3*(wp3[0]-wp1[0]))/(rxy3p(wp1,wp3)))
    ap1y = ((Gp*mp2*(wp2[1]-wp1[1]))/(rxy3p(wp1,wp2))) + ((Gp*mp3*(wp3[1]-wp1[1]))/(rxy3p(wp1,wp3)))
    jp = np.array([wp1[2],wp1[3],ap1x,ap1y])
    return jp

def RFp2(wp1,wp2,wp3):
    ap2x = (-(Gp*mp1*(wp2[0]-wp1[0]))/(rxy3p(wp1,wp2))) + ((Gp*mp3*(wp3[0]-wp2[0]))/(rxy3p(wp2,wp3)))
    ap2y = (-(Gp*mp1*(wp2[1]-wp1[1]))/(rxy3p(wp1,wp2))) + ((Gp*mp3*(wp3[1]-wp2[1]))/(rxy3p(wp2,wp3)))
    jp = np.array([wp2[2],wp2[3],ap2x,ap2y])
    return jp

def RFp3(wp1,wp2,wp3):
    ap3x = (-(Gp*mp1*(wp3[0]-wp1[0]))/(rxy3p(wp1,wp3))) + (-(Gp*mp2*(wp3[0]-wp2[0]))/(rxy3p(wp2,wp3)))
    ap3y = (-(Gp*mp1*(wp3[1]-wp1[1]))/(rxy3p(wp1,wp3))) + (-(Gp*mp2*(wp3[1]-wp2[1]))/(rxy3p(wp2,wp3)))
    jp = np.array([wp3[2],wp3[3],ap3x,ap3y])
    return jp

def RFp(wp):
    wp1, wp2, wp3 = np.split(wp,3)
    jp1 = RFp1(wp1,wp2,wp3)
    jp2 = RFp2(wp1,wp2,wp3)
    jp3 = RFp3(wp1,wp2,wp3)
    return np.concatenate((jp1,jp2,jp3))

def RKp4(wp):
    kp1 = RFp(wp)
    kp2 = RFp(wp + 0.5*kp1*hp)
    kp3 = RFp(wp + 0.5*kp2*hp)
    kp4 = RFp(wp + kp3*hp)
    RKp4 = hp*(1/6)*(kp1 + 2*kp2 + 2*kp3 + kp4)
    return RKp4

while n2 < nstepsp:
    wp = wp + RKp4(wp)
    rp1x_list.append(wp[0])
    rp1y_list.append(wp[1])
    vp1x_list.append(wp[2])
    vp1y_list.append(wp[3])
    rp2x_list.append(wp[4])
    rp2y_list.append(wp[5])
    vp2x_list.append(wp[6])
    vp2y_list.append(wp[7])
    rp3x_list.append(wp[8])
    rp3y_list.append(wp[9])
    vp3x_list.append(wp[10])
    vp3y_list.append(wp[11])
    tp = tp + hp 
    tp_list.append(tp)
    n2 = n2 + 1

#plt.figure(10)
#plt.plot(rp1x_list,rp1y_list,color = "red",linewidth = 1)
#plt.plot(rp2x_list,rp2y_list,color = "green", linewidth = 1)
#plt.plot(rp3x_list,rp3y_list,color = "blue", linewidth = 1)
#plt.grid(True)
#plt.axis("scaled")  
#plt.xlim(-k,k)
#plt.ylim(-k,k)


print(nstepsp)
min_distance_list = []
distance_list_new = [] 
i = 0
while i < nstepsp:
    #w is perturbed solution
    w = np.array([r1x_list[i], r1y_list[i], v1x_list[i], v1y_list[i], r2x_list[i], r2y_list[i], v2x_list[i], v2y_list[i], r3x_list[i], r3y_list[i], v3x_list[i], v3y_list[i]])
    j = 0
    while j < number_of_timesteps:
        #wp is original solution
        wp  = np.array([rp1x_list[j], rp1y_list[j], vp1x_list[j], vp1y_list[j], rp2x_list[j], rp2y_list[j], vp2x_list[j], vp2y_list[j], rp3x_list[j], rp3y_list[j], vp3x_list[j], vp3y_list[j]])
        distance =  w_distance(w[0], wp[0], w[1], wp[1], w[2], wp[2], w[3], wp[3],w[4], wp[4], w[5], wp[5], w[6], wp[6], w[7], wp[7], w[8], wp[8], w[9], wp[9], w[10], wp[10], w[11], wp[11])  
        distance_list_new.append(distance)
        if j == number_of_timesteps-1:
            min_distance = min(distance_list_new)
            min_distance_list.append(min_distance)
            distance_list_new.clear()
        j = j + 1
    i = i + 1

plt.figure(14)
plt.title("Orbital Difference")
plt.plot(t_list, min_distance_list, color = "red")
plt.grid(True)
plt.show()

orbital_difference = max(min_distance_list)
print("orbital difference = ", orbital_difference)
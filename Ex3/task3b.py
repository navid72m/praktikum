from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



def function (t,y):
    return ([y[0]* -y[1]-y[0]*((y[0]**2)+y[1])**2, y[0]+ y[1]-y[1]*((y[0]**2)+y[1])**2])

sol = solve_ivp(function,[0, 20],[2,0],dense_output=True,vectorized=True   )

print(sol.t)
print(sol.y)

plt.scatter(sol.y[0],sol.y[1], s=20, edgecolors='none', c='green')
plt.show()

sol1 = solve_ivp(function,[0, 20],[0.5,0],dense_output=True,vectorized=True   )

print(sol.t)
print(sol.y)

plt.scatter(sol1.y[0],sol1.y[1], s=20, edgecolors='none', c='green')
plt.show()
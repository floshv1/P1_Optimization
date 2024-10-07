# P1_Optimization
A company wants to maximize its profit function given by:

f(x,y)=−x²+4xy−2y²

where x and y represent the amount of resources allocated to two different projects.

The problem is subject to the following constraints:

The total resources used must not exceed 30 units:
   
x+2y≤30

The product of the resources allocated must be at least 50 units:
   
xy≥50

The relationship between the resource allocations must satisfy the following nonlinear constraint:
   
y≤(3x²/100)+5

The company wants to find the values of xand y that maximize the profit f(x,y) under these constraints. Before running calculations, though, can you help the company drawing the problem in a graph? Can you identify the feasible region? Is the region convex?

Submit the link to a python code (ideally within your github repo or a colab file) solving the general problem and testing this particular proposed question using the Sequential Least SQares Programming optimizer (SLSQP), as implemented in scipy. The program should show:

The possibility of inputting the values of the bounds of the independent variables as arguments. Provide a help argument that yields the format for the command line execution of the code with arguments.
A plot of the objective funcion and its feasible region.
The solution to the proposed problem.
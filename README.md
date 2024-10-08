## P1_Optimization Florian CAZAC

A company wants to maximize its profit function given by:

* ***f(x,y)=−x²+4xy−2y²***

where x and y represent the amount of resources allocated to two different projects.

**The problem is subject to the following constraints:**

The total resources used must not exceed 30 units:
   
* ***x+2y≤30***

The product of the resources allocated must be at least 50 units:
   
* ***xy≥50***

The relationship between the resource allocations must satisfy the following nonlinear constraint:
   
* ***y≤(3x²/100)+5***

The company wants to find the values of xand y that maximize the profit f(x,y) under these constraints. Before running calculations, though, can you help the company drawing the problem in a graph? Can you identify the feasible region? Is the region convex?

Submit the link to a python code (ideally within your github repo or a colab file) solving the general problem and testing this particular proposed question using the Sequential Least SQares Programming optimizer (SLSQP), as implemented in scipy. 

### The program should show:
***1/ The possibility of inputting the values of the bounds of the independent variables as arguments. Provide a help argument that yields the format for the command line execution of the code with arguments.***

**Help Command :**
> You need to put the following command in your shell and in the good folder :
> 
> * python your_script.py --help
>
> **Output :**
>> SLSQP Optimization for Resource Allocation
>>
>> options:<br>
-h, --help   show this help message and exit;<br>
--xmin XMIN  Minimum bound for x;<br>
--xmax XMAX  Maximum bound for x;<br>
--ymin YMIN  Minimum bound for y;<br>
--ymax YMAX  Maximum bound for y;<br>
>>
>> Use the following format: python your_script.py --xmin <value> --xmax <value> --ymin <value> --ymax <value>

**Run code with alternative bounds :**

> Use Following format to run the code with differents values of the bounds of the independent variables as arguments is : 
> 
> * python your_script.py --xmin <value> --xmax <value> --ymin <value> --ymax <value>
>
> *Example : python P1_Florian_Cazac.py --xmin 1 --xmax 25 --ymin 1 --ymax 20*
> 
>*By default : xmin = 0, xmax = 30, ymin = 0, y max = 30*

***2/ A plot of the objective funcion and its feasible region.***

***3/ The solution to the proposed problem.***

